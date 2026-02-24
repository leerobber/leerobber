"""SEO optimization agent - optimizes content for search engines."""

from contentai_pro.ai.agents.base import AgentResult, BaseAgent
from contentai_pro.core.config import settings
from contentai_pro.modules.content.templates import seo_prompt


class SEOAgent(BaseAgent):
    name = "seo"

    async def execute(self, content: str, keywords: list[str]) -> AgentResult:
        prompt = seo_prompt(content, keywords)
        output = await self._call_llm(prompt, settings.default_model)
        return AgentResult(agent_name=self.name, output=output)
