from pydantic import BaseModel


class ContentRequest(BaseModel):
    email: str
    password: str
    topic: str
    keywords: list[str]


class ContentResponse(BaseModel):
    content: str
    credits_remaining: int
    quality_score: str = "Excellent"
    pipeline_stages: list[str] = []


class StreamEvent(BaseModel):
    event: str
    stage: str
    content: str = ""
    progress: float = 0.0
