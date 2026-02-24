"""Writer agent - drafts content based on research notes."""

from contentai_pro.ai.agents.base import AgentResult, BaseAgent
from contentai_pro.core.config import settings
from contentai_pro.modules.content.templates import writing_prompt


class WriterAgent(BaseAgent):
    name = "writer"

    async def execute(
        self, topic: str, keywords: list[str], research_notes: str
    ) -> AgentResult:
        prompt = writing_prompt(topic, keywords, research_notes)
        output = await self._call_llm(prompt, settings.default_model)
        return AgentResult(agent_name=self.name, output=output)
