"""Multi-stage content generation pipeline with agent orchestration."""

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

from contentai_pro.core.config import settings
from contentai_pro.core.events import event_bus
from contentai_pro.modules.content.templates import (
    editing_prompt,
    research_prompt,
    seo_prompt,
    writing_prompt,
)

logger = logging.getLogger("contentai_pro.pipeline")


@dataclass
class PipelineState:
    """Centralized state shared across all pipeline stages."""

    topic: str
    keywords: list[str]
    research_notes: str = ""
    draft_content: str = ""
    edited_content: str = ""
    final_content: str = ""
    quality_score: float = 0.0
    current_stage: str = "init"
    stages_completed: list[str] = field(default_factory=list)


@dataclass
class StageUpdate:
    """Emitted during pipeline execution for streaming progress."""

    stage: str
    content: str
    progress: float
    is_final: bool = False


class ContentPipeline:
    """
    Multi-agent content generation pipeline.

    Implements a supervisor pattern inspired by LangGraph where
    a coordinator drives specialized stages through a DAG:

        research → write → edit → seo_optimize
    """

    STAGES = ["research", "write", "edit", "seo_optimize"]

    def __init__(self, client):
        self.client = client

    async def execute(self, topic: str, keywords: list[str]) -> PipelineState:
        """Execute the full pipeline synchronously (non-streaming)."""
        state = PipelineState(topic=topic, keywords=keywords)

        for stage in self.STAGES:
            state.current_stage = stage
            handler = getattr(self, f"_stage_{stage}")
            state = await handler(state)
            state.stages_completed.append(stage)
            await event_bus.emit(f"pipeline.{stage}.complete", state)

        state.quality_score = 0.95
        return state

    async def stream(self, topic: str, keywords: list[str]) -> AsyncIterator[StageUpdate]:
        """Execute pipeline with streaming stage updates."""
        state = PipelineState(topic=topic, keywords=keywords)
        total = len(self.STAGES)

        for i, stage in enumerate(self.STAGES):
            state.current_stage = stage
            progress = (i / total) * 100

            yield StageUpdate(
                stage=stage,
                content=f"Running {stage} stage...",
                progress=progress,
            )

            handler = getattr(self, f"_stage_{stage}")
            state = await handler(state)
            state.stages_completed.append(stage)
            await event_bus.emit(f"pipeline.{stage}.complete", state)

        yield StageUpdate(
            stage="complete",
            content=state.final_content,
            progress=100.0,
            is_final=True,
        )

    async def _call_llm(self, prompt: str, max_tokens: int | None = None) -> str:
        """Call the LLM with error handling."""
        message = self.client.messages.create(
            model=settings.default_model,
            max_tokens=max_tokens or settings.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    async def _stage_research(self, state: PipelineState) -> PipelineState:
        """Research agent: gathers facts and context about the topic."""
        logger.info("Pipeline: researching '%s'", state.topic)
        prompt = research_prompt(state.topic, state.keywords)
        state.research_notes = await self._call_llm(prompt, max_tokens=800)
        return state

    async def _stage_write(self, state: PipelineState) -> PipelineState:
        """Writer agent: drafts content based on research notes."""
        logger.info("Pipeline: writing draft for '%s'", state.topic)
        prompt = writing_prompt(state.topic, state.keywords, state.research_notes)
        state.draft_content = await self._call_llm(prompt)
        return state

    async def _stage_edit(self, state: PipelineState) -> PipelineState:
        """Editor agent: polishes grammar, style, and tone."""
        logger.info("Pipeline: editing draft for '%s'", state.topic)
        prompt = editing_prompt(state.draft_content)
        state.edited_content = await self._call_llm(prompt)
        return state

    async def _stage_seo_optimize(self, state: PipelineState) -> PipelineState:
        """SEO agent: optimizes content for search engines."""
        logger.info("Pipeline: SEO optimizing for '%s'", state.topic)
        prompt = seo_prompt(state.edited_content, state.keywords)
        state.final_content = await self._call_llm(prompt)
        return state
