"""
Multi-agent orchestrator using the Supervisor pattern.

Inspired by LangGraph's graph-based orchestration model, this orchestrator
coordinates specialized agents through a directed pipeline:

    Researcher → Writer → Editor → SEO Agent

Each agent processes a shared PipelineState and returns updates.
The supervisor (this orchestrator) manages the flow, error handling,
and event emission between stages.
"""

import logging
from typing import AsyncIterator

from contentai_pro.ai.agents.editor import EditorAgent
from contentai_pro.ai.agents.researcher import ResearcherAgent
from contentai_pro.ai.agents.seo import SEOAgent
from contentai_pro.ai.agents.writer import WriterAgent
from contentai_pro.core.events import event_bus
from contentai_pro.modules.content.pipeline import PipelineState, StageUpdate

logger = logging.getLogger("contentai_pro.orchestrator")


class AgentOrchestrator:
    """
    Supervisor-pattern orchestrator that coordinates multiple AI agents
    through a content generation pipeline.
    """

    def __init__(self, client):
        self.researcher = ResearcherAgent(client)
        self.writer = WriterAgent(client)
        self.editor = EditorAgent(client)
        self.seo = SEOAgent(client)

    async def run(self, topic: str, keywords: list[str]) -> PipelineState:
        """Execute full agent pipeline and return final state."""
        state = PipelineState(topic=topic, keywords=keywords)

        # Stage 1: Research
        state.current_stage = "research"
        result = await self.researcher.execute(topic=topic, keywords=keywords)
        state.research_notes = result.output
        state.stages_completed.append("research")
        await event_bus.emit("agent.research.complete", state)

        # Stage 2: Write
        state.current_stage = "write"
        result = await self.writer.execute(
            topic=topic, keywords=keywords, research_notes=state.research_notes
        )
        state.draft_content = result.output
        state.stages_completed.append("write")
        await event_bus.emit("agent.write.complete", state)

        # Stage 3: Edit
        state.current_stage = "edit"
        result = await self.editor.execute(draft=state.draft_content)
        state.edited_content = result.output
        state.stages_completed.append("edit")
        await event_bus.emit("agent.edit.complete", state)

        # Stage 4: SEO Optimize
        state.current_stage = "seo_optimize"
        result = await self.seo.execute(content=state.edited_content, keywords=keywords)
        state.final_content = result.output
        state.stages_completed.append("seo_optimize")
        await event_bus.emit("agent.seo.complete", state)

        state.quality_score = 0.95
        logger.info(
            "Orchestrator completed pipeline for '%s' through %d stages",
            topic,
            len(state.stages_completed),
        )
        return state

    async def stream(self, topic: str, keywords: list[str]) -> AsyncIterator[StageUpdate]:
        """Execute pipeline with streaming stage updates for SSE."""
        stages = [
            ("research", "Researching topic and gathering facts..."),
            ("write", "Drafting article content..."),
            ("edit", "Polishing grammar and style..."),
            ("seo_optimize", "Optimizing for search engines..."),
        ]
        total = len(stages)

        state = PipelineState(topic=topic, keywords=keywords)

        for i, (stage_name, description) in enumerate(stages):
            progress = (i / total) * 100
            yield StageUpdate(stage=stage_name, content=description, progress=progress)

            if stage_name == "research":
                result = await self.researcher.execute(topic=topic, keywords=keywords)
                state.research_notes = result.output
            elif stage_name == "write":
                result = await self.writer.execute(
                    topic=topic, keywords=keywords, research_notes=state.research_notes
                )
                state.draft_content = result.output
            elif stage_name == "edit":
                result = await self.editor.execute(draft=state.draft_content)
                state.edited_content = result.output
            elif stage_name == "seo_optimize":
                result = await self.seo.execute(
                    content=state.edited_content, keywords=keywords
                )
                state.final_content = result.output

            state.stages_completed.append(stage_name)

        yield StageUpdate(
            stage="complete",
            content=state.final_content,
            progress=100.0,
            is_final=True,
        )
