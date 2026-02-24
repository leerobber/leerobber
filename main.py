"""
ContentAI Pro - AI-Powered Content Generation Platform

Production-grade FastAPI application with:
- Multi-agent AI pipeline (Research → Write → Edit → SEO)
- SSE streaming for real-time content generation
- Event-driven architecture for decoupled module communication
- Modular monolith structure with domain separation
- RAG-ready hybrid retrieval architecture
"""

import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from contentai_pro.core.config import settings
from contentai_pro.core.middleware import LoggingMiddleware, RequestIDMiddleware
from contentai_pro.modules.content.router import router as content_router
from contentai_pro.modules.contact.router import router as contact_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("contentai_pro")


def create_app() -> FastAPI:
    """Application factory - creates and configures the FastAPI app."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="AI-powered content generation with multi-agent pipeline orchestration",
    )

    # Middleware (registered in reverse order - last registered runs first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Register module routers
    app.include_router(content_router)
    app.include_router(contact_router)

    # Mount static files
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Homepage - serve static HTML
    @app.get("/")
    async def homepage():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
        return {"message": f"{settings.app_name} v{settings.version} is running"}

    # Legacy endpoint compatibility - redirect old /generate to new API
    from contentai_pro.modules.content.models import ContentRequest
    from contentai_pro.modules.content.router import generate_quick

    @app.post("/generate")
    async def legacy_generate(request: ContentRequest):
        return await generate_quick(request)

    # Health check
    @app.get("/health")
    async def health():
        return {
            "status": "live",
            "service": settings.app_name,
            "version": settings.version,
            "features": [
                "multi_agent_pipeline",
                "sse_streaming",
                "event_driven",
                "rag_ready",
                "modular_architecture",
            ],
        }

    logger.info(
        "%s v%s started | API key configured: %s",
        settings.app_name,
        settings.version,
        bool(settings.anthropic_api_key),
    )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=settings.debug)
