"""Content generation service - coordinates pipeline execution and credit management."""

import logging

import anthropic

from contentai_pro.core.config import settings
from contentai_pro.core.events import event_bus
from contentai_pro.modules.auth.service import auth_service
from contentai_pro.modules.content.models import ContentResponse
from contentai_pro.modules.content.pipeline import ContentPipeline

logger = logging.getLogger("contentai_pro.content")


class ContentService:
    def __init__(self):
        if settings.anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self.pipeline = ContentPipeline(self.client)
        else:
            self.client = None
            self.pipeline = None

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    async def generate(self, email: str, topic: str, keywords: list[str]) -> ContentResponse:
        """Generate content through the full multi-agent pipeline."""
        state = await self.pipeline.execute(topic, keywords)
        remaining = auth_service.deduct_credit(email)

        await event_bus.emit("content.generated", {
            "email": email,
            "topic": topic,
            "keywords": keywords,
        })

        return ContentResponse(
            content=state.final_content,
            credits_remaining=remaining,
            pipeline_stages=state.stages_completed,
        )

    async def generate_simple(self, email: str, topic: str, keywords: list[str]) -> ContentResponse:
        """Quick single-pass generation (no multi-stage pipeline)."""
        prompt = f"""Write a professional, engaging, and valuable 700-word article about: {topic}

Keywords to include naturally throughout: {', '.join(keywords)}

Requirements:
- Start with a compelling hook that grabs attention
- Include 3-4 well-developed main points with subheadings
- Add real-world examples and actionable insights
- Use a professional yet conversational tone
- End with a strong conclusion and clear takeaway
- Optimize for SEO without keyword stuffing

Write content that provides real value to readers."""

        message = self.client.messages.create(
            model=settings.default_model,
            max_tokens=settings.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        content = message.content[0].text
        remaining = auth_service.deduct_credit(email)

        await event_bus.emit("content.generated", {
            "email": email,
            "topic": topic,
        })

        return ContentResponse(content=content, credits_remaining=remaining)


content_service = ContentService()
