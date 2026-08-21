"""FastAPI application for the local Perception Workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roadsense import __version__
from roadsense.api.models import DemoResponse, HealthResponse, ReadinessResponse, ReportResponse
from roadsense.evidence import build_fixture_report
from roadsense.fixture import build_demo_payload


def create_app() -> Any:
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - exercised by fresh base-wheel imports
        raise ImportError("serve requires roadsense-perception[serve]") from exc

    app = FastAPI(title="RoadSense-Perception", version=__version__)
    web_dir = Path(__file__).resolve().parent.parent / "web"
    # Keep the explicit /static mount for callers that want a namespaced asset
    # URL. The root mount is registered after API routes below so the local
    # browser page can resolve its relative Pages-compatible asset paths.

    @app.get("/", include_in_schema=False)
    async def index() -> Any:
        return FileResponse(web_dir / "index.html")

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "roadsense-perception",
            "version": __version__,
            "runtime": "deterministic_geometric_fixture",
        }

    @app.get("/api/v1/readiness", response_model=ReadinessResponse)
    async def readiness() -> dict[str, object]:
        return {
            "status": "ready",
            "verification_level": "fixture",
            "model_loaded": False,
            "benchmark_claim_available": False,
        }

    @app.get("/api/v1/demo", response_model=DemoResponse)
    async def demo() -> dict[str, object]:
        return build_demo_payload()

    @app.get("/api/v1/report", response_model=ReportResponse)
    async def report() -> dict[str, object]:
        return build_fixture_report()

    app.mount("/static", StaticFiles(directory=web_dir), name="static")
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app
