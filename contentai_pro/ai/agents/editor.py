"""Editor agent - polishes grammar, style, and tone."""

from contentai_pro.ai.agents.base import AgentResult, BaseAgent
from contentai_pro.core.config import settings
from contentai_pro.modules.content.templates import editing_prompt


class EditorAgent(BaseAgent):
    name = "editor"

    async def execute(self, draft: str) -> AgentResult:
        prompt = editing_prompt(draft)
        output = await self._call_llm(prompt, settings.default_model)
        return AgentResult(agent_name=self.name, output=output)
