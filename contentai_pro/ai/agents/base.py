"""Base agent interface for the multi-agent content pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentResult:
    """Standardized result from any agent execution."""

    agent_name: str
    output: str
    metadata: dict[str, Any] | None = None


class BaseAgent(ABC):
    """Abstract base for all pipeline agents."""

    name: str = "base"

    def __init__(self, client):
        self.client = client

    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the agent's task and return a result."""

    async def _call_llm(self, prompt: str, model: str, max_tokens: int = 1500) -> str:
        message = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
