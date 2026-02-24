"""Research agent - gathers facts and context for content generation."""

from contentai_pro.ai.agents.base import AgentResult, BaseAgent
from contentai_pro.core.config import settings
from contentai_pro.modules.content.templates import research_prompt


class ResearcherAgent(BaseAgent):
    name = "researcher"

    async def execute(self, topic: str, keywords: list[str]) -> AgentResult:
        prompt = research_prompt(topic, keywords)
        output = await self._call_llm(prompt, settings.default_model, max_tokens=800)
        return AgentResult(agent_name=self.name, output=output)
