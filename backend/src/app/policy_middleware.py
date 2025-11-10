"""
Policy enforcement middleware for PromptSail.

Integrates IronClad policy engine to enforce security policies
before forwarding requests to LLM providers.
"""

from typing import Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import json
import os
import sys

# Import IronClad policy engine
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../ironclad_policies"))

from ironclad_policies import PolicyEngine, PolicyDecision
from ironclad_policies.content import ContentPolicy
from ironclad_policies.content_lite import ContentPolicyLite
from ironclad_policies.models import ContentFilterConfig, PolicyAction
from ironclad_policies.config_manager import ConfigManager


class PolicyEnforcer:
    """
    Enforces IronClad security policies on LLM requests.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize policy enforcer.

        Args:
            config_file: Path to YAML config (default: use env vars)
        """
        self.engine = PolicyEngine()

        # Load configuration
        if config_file:
            config_mgr = ConfigManager()
            config = config_mgr.load_config(config_file)
        else:
            # Default config from environment
            config = self._create_default_config()

        # Add policy based on mode
        mode = os.getenv("IRONCLAD_MODE", "full")
        if mode == "lite":
            policy = ContentPolicyLite(config=config)
            print("🚀 IronClad Policy Engine initialized (mode: LITE)")
        else:
            policy = ContentPolicy(config=config)
            print("🚀 IronClad Policy Engine initialized (mode: FULL)")

        self.engine.add_policy(policy)

        summary = self.engine.get_policy_summary()
        print(f"   Policies active: {summary['total_policies']}")
        print(f"   Policy types: {', '.join(summary['policy_types'])}")

    def _create_default_config(self) -> ContentFilterConfig:
        """Create default config from environment variables."""
        return ContentFilterConfig(
            enabled=os.getenv("IRONCLAD_ENABLED", "true").lower() == "true",
            detect_pii=os.getenv("DETECT_PII", "true").lower() == "true",
            detect_phi=os.getenv("DETECT_PHI", "false").lower() == "true",
            detect_pci=os.getenv("DETECT_PCI", "false").lower() == "true",
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.7")),
            default_action=PolicyAction(os.getenv("DEFAULT_ACTION", "warn")),
        )

    async def evaluate_request(
        self,
        request_body: dict,
        metadata: Optional[dict] = None
    ) -> PolicyDecision:
        """
        Evaluate LLM request against policies.

        Args:
            request_body: Parsed JSON request body
            metadata: Additional context (project, user, etc.)

        Returns:
            PolicyDecision with allowed/blocked status
        """
        # Evaluate using engine's built-in request parsing
        decision = await self.engine.evaluate_request_body(
            request_body=request_body,
            metadata=metadata
        )

        return decision

    def create_error_response(self, decision: PolicyDecision) -> JSONResponse:
        """
        Create HTTP 403 error response for blocked requests.

        Args:
            decision: PolicyDecision that resulted in BLOCK

        Returns:
            JSONResponse with structured error
        """
        # Format violations for user-friendly display
        violations_summary = []
        for violation in decision.violations:
            violations_summary.append({
                "type": violation.policy_type,
                "severity": violation.severity,
                "message": violation.message,
                "detections": len(violation.detections),
            })

        error_body = {
            "error": {
                "message": "Request blocked by IronClad-AI security policy",
                "type": "policy_violation",
                "code": "IRONCLAD_POLICY_BLOCK",
                "details": {
                    "action": decision.action.value,
                    "violations": violations_summary,
                    "total_violations": len(decision.violations),
                    "help": "Please remove sensitive information and try again. "
                            "Contact your administrator if you believe this is an error."
                }
            }
        }

        return JSONResponse(
            status_code=403,
            content=error_body
        )


# Global enforcer instance
_policy_enforcer: Optional[PolicyEnforcer] = None


def get_policy_enforcer() -> PolicyEnforcer:
    """Get or create the global policy enforcer instance."""
    global _policy_enforcer
    if _policy_enforcer is None:
        profile = os.getenv("IRONCLAD_PROFILE", "default")
        _policy_enforcer = PolicyEnforcer(config_file=profile)
    return _policy_enforcer


async def enforce_policy(
    request: Request,
    request_body: dict,
    project_slug: str,
    provider_slug: str
) -> tuple[bool, Optional[dict], Optional[JSONResponse]]:
    """
    Enforce policy on LLM request.

    Args:
        request: FastAPI Request object
        request_body: Parsed JSON request body
        project_slug: PromptSail project identifier
        provider_slug: LLM provider identifier

    Returns:
        Tuple of (allowed, modified_body, error_response)
        - allowed: True if request should proceed
        - modified_body: Modified request body if REDACT action
        - error_response: JSONResponse if BLOCK action
    """
    # Check if policies are enabled
    if os.getenv("IRONCLAD_ENABLED", "true").lower() != "true":
        return True, None, None

    # Get enforcer
    enforcer = get_policy_enforcer()

    # Build metadata
    metadata = {
        "project": project_slug,
        "provider": provider_slug,
        "ip_address": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
    }

    # Evaluate request
    decision = await enforcer.evaluate_request(request_body, metadata)

    # Handle decision
    if decision.action == PolicyAction.BLOCK:
        # Return 403 error
        error_response = enforcer.create_error_response(decision)
        return False, None, error_response

    elif decision.action == PolicyAction.REDACT:
        # Modify request body with redacted content
        modified_body = request_body.copy()

        # Replace content in messages (OpenAI format)
        if "messages" in modified_body and decision.modified_content:
            modified_body["messages"][-1]["content"] = decision.modified_content

        # Replace prompt (Anthropic format)
        if "prompt" in modified_body and decision.modified_content:
            modified_body["prompt"] = decision.modified_content

        return True, modified_body, None

    else:
        # ALLOW or WARN - proceed unchanged
        return True, None, None
