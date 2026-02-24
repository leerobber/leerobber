"""Content generation API endpoints."""

import json

from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

from contentai_pro.core.dependencies import require_credits
from contentai_pro.modules.content.models import ContentRequest, ContentResponse
from contentai_pro.modules.content.service import content_service

router = APIRouter(prefix="/api/content", tags=["content"])


@router.post("/generate", response_model=ContentResponse)
async def generate_content(request: ContentRequest):
    """Generate content using the multi-agent pipeline."""
    if not content_service.is_configured:
        raise HTTPException(status_code=500, detail="API key not configured")

    user = await require_credits(request.email, request.password)

    return await content_service.generate(
        email=request.email,
        topic=request.topic,
        keywords=request.keywords,
    )


@router.post("/generate/quick", response_model=ContentResponse)
async def generate_quick(request: ContentRequest):
    """Quick single-pass content generation (faster, less refined)."""
    if not content_service.is_configured:
        raise HTTPException(status_code=500, detail="API key not configured")

    user = await require_credits(request.email, request.password)

    return await content_service.generate_simple(
        email=request.email,
        topic=request.topic,
        keywords=request.keywords,
    )


@router.post("/generate/stream")
async def generate_stream(request: ContentRequest):
    """Stream content generation with real-time stage updates via SSE."""
    if not content_service.is_configured:
        raise HTTPException(status_code=500, detail="API key not configured")

    user = await require_credits(request.email, request.password)

    async def event_generator():
        async for update in content_service.pipeline.stream(
            request.topic, request.keywords
        ):
            data = json.dumps({
                "stage": update.stage,
                "content": update.content,
                "progress": update.progress,
            })
            yield f"event: {update.stage}\ndata: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
