"""
Health check endpoint with IronClad policy engine status.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from app.policy_middleware import get_policy_enforcer
import os

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check with policy engine status.

    Returns:
        JSONResponse with service health status including:
        - Overall health status
        - Timestamp
        - Policy engine status (if enabled)
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {}
    }

    # Check if IronClad is enabled
    if os.getenv("IRONCLAD_ENABLED", "true").lower() == "true":
        # Check Policy Engine
        try:
            enforcer = get_policy_enforcer()
            summary = enforcer.engine.get_policy_summary()
            health["services"]["policy_engine"] = {
                "status": "healthy",
                "mode": os.getenv("IRONCLAD_MODE", "full"),
                "policies_loaded": summary["total_policies"],
                "policy_types": summary["policy_types"],
                "profile": os.getenv("IRONCLAD_PROFILE", "default")
            }
        except Exception as e:
            health["services"]["policy_engine"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"
    else:
        health["services"]["policy_engine"] = {
            "status": "disabled",
            "message": "IronClad policy enforcement is disabled"
        }

    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(health, status_code=status_code)


@router.get("/ironclad/health")
async def ironclad_health():
    """
    Detailed IronClad policy engine health check.

    Returns:
        JSONResponse with detailed policy engine configuration and status
    """
    if os.getenv("IRONCLAD_ENABLED", "true").lower() != "true":
        return JSONResponse({
            "enabled": False,
            "message": "IronClad policy enforcement is disabled"
        })

    try:
        enforcer = get_policy_enforcer()
        summary = enforcer.engine.get_policy_summary()

        config = {
            "enabled": True,
            "mode": os.getenv("IRONCLAD_MODE", "full"),
            "profile": os.getenv("IRONCLAD_PROFILE", "default"),
            "detection": {
                "pii": os.getenv("DETECT_PII", "true").lower() == "true",
                "phi": os.getenv("DETECT_PHI", "false").lower() == "true",
                "pci": os.getenv("DETECT_PCI", "false").lower() == "true"
            },
            "min_confidence": float(os.getenv("MIN_CONFIDENCE", "0.7")),
            "default_action": os.getenv("DEFAULT_ACTION", "warn"),
            "policies": {
                "total": summary["total_policies"],
                "types": summary["policy_types"]
            }
        }

        return JSONResponse(config, status_code=200)
    except Exception as e:
        return JSONResponse({
            "enabled": True,
            "status": "error",
            "error": str(e)
        }, status_code=503)
