import os
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "ContentAI Pro"
    version: str = "3.0"
    debug: bool = False
    anthropic_api_key: str = os.environ.get("ANTHROPIC_KEY", "")
    default_model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 2000
    default_credits: int = 15
    cors_origins: list[str] = ["*"]


settings = Settings()
