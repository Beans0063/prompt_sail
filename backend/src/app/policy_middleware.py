"""
Policy enforcement middleware for PromptSail.

Integrates IronClad policy engine to enforce security policies
before forwarding requests to LLM providers.
"""

from typing import Optional, Any
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import json
import os
import sys

# Import IronClad policy engine
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../ironclad_policies"))

from ironclad_policies import PolicyEngine, PolicyDecision
from ironclad_policies.content import ContentPolicy
from ironclad_policies.content_lite import ContentPolicyLite
from ironclad_policies.models import ContentFilterConfig, PolicyAction, SensitiveDataType

# Try to import LLM Guard policy (optional dependency)
try:
    from ironclad_policies.llm_guard_policy import LLMGuardPolicy, LLM_GUARD_AVAILABLE
except ImportError:
    LLM_GUARD_AVAILABLE = False
    LLMGuardPolicy = None


class PolicyEnforcer:
    """
    Enforces IronClad security policies on LLM requests.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize policy enforcer.

        Args:
            config_file: Path to YAML config (default: use env vars) - currently unused
        """
        self.engine = PolicyEngine()

        # Create configuration from environment variables
        config = self._create_default_config()

        # Add policy based on mode
        mode = os.getenv("IRONCLAD_MODE", "full")
        if mode == "lite":
            policy = ContentPolicyLite(config=config)
            self.engine.add_policy(policy)
            print("🚀 IronClad Policy Engine initialized (mode: LITE)")
        else:
            # FULL mode: Add ContentPolicy + LLMGuardPolicy (if available)
            # Try ContentPolicy first, fall back to ContentPolicyLite if NLP fails
            try:
                content_policy = ContentPolicy(config=config)
                self.engine.add_policy(content_policy)
                print("   ✅ Content policy loaded (with NLP)")
            except Exception as e:
                print(f"   ⚠️  ContentPolicy failed ({str(e)[:100]}), using regex-only mode")
                content_policy = ContentPolicyLite(config=config)
                self.engine.add_policy(content_policy)
                print("   ✅ Content policy loaded (regex-only)")

            # Try to add LLM Guard policy for adversarial protection
            if LLM_GUARD_AVAILABLE:
                try:
                    llm_guard_config = self._load_llm_guard_config()
                    llm_guard_policy = LLMGuardPolicy(config=llm_guard_config)
                    self.engine.add_policy(llm_guard_policy)
                    print("🚀 IronClad Policy Engine initialized (mode: FULL)")
                    print("   ✅ LLM Guard enabled (adversarial protection)")
                except Exception as e:
                    print(f"   ⚠️  LLM Guard initialization failed: {e}")
                    print("   Continuing with Content-only protection")
            else:
                print("🚀 IronClad Policy Engine initialized (mode: FULL)")
                print("   ⚠️  LLM Guard not available (install with: pip install llm-guard)")

        summary = self.engine.get_policy_summary()
        print(f"   Policies active: {summary['total_policies']}")
        print(f"   Policy types: {', '.join(summary['policy_types'])}")

    def _create_default_config(self) -> ContentFilterConfig:
        """Create default config from environment variables."""
        # Configure action overrides for specific data types
        action_overrides = {
            SensitiveDataType.SSN: PolicyAction.BLOCK,  # Block SSNs
            SensitiveDataType.EMAIL: PolicyAction.ALLOW,  # Let LLM Guard handle emails with vault
            SensitiveDataType.LOCATION: PolicyAction.WARN,  # Warn on locations
        }

        return ContentFilterConfig(
            enabled=os.getenv("IRONCLAD_ENABLED", "true").lower() == "true",
            detect_pii=os.getenv("DETECT_PII", "true").lower() == "true",
            detect_phi=os.getenv("DETECT_PHI", "false").lower() == "true",
            detect_pci=os.getenv("DETECT_PCI", "false").lower() == "true",
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.7")),
            default_action=PolicyAction(os.getenv("DEFAULT_ACTION", "warn")),
            action_overrides=action_overrides,
        )

    def _load_llm_guard_config(self) -> dict:
        """Load LLM Guard configuration from YAML file."""
        import yaml

        # Try to load from security profile or direct config path
        security_profile = os.getenv("IRONCLAD_SECURITY_PROFILE", None)

        if security_profile:
            # Load from security_profiles directory
            config_path = f"/app/ironclad_policies/config/security_profiles/{security_profile}.yaml"
            print(f"   Loading LLM Guard config from security profile: {security_profile}")
        else:
            # Load from direct path
            config_path = os.getenv("LLM_GUARD_CONFIG", "/app/ironclad_policies/config/llm_guard.yaml")

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get('llm_guard', {})
        except Exception as e:
            print(f"   Warning: Could not load LLM Guard config from {config_path}: {e}")

        # Return default minimal config
        return {
            "enabled": True,
            "input_scanners": {
                "prompt_injection": {
                    "enabled": True,
                    "threshold": 0.75,
                    "use_onnx": True
                },
                "secrets": {
                    "enabled": True,
                    "redact_mode": "all"
                },
                "toxicity": {
                    "enabled": True,
                    "threshold": 0.7
                },
                "invisible_text": {
                    "enabled": True
                }
            },
            "output_scanners": {}
        }

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

    async def evaluate_output(
        self,
        prompt: str,
        output: str,
        metadata: Optional[dict] = None
    ) -> PolicyDecision:
        """
        Evaluate LLM output against policies.

        Args:
            prompt: Original user prompt
            output: LLM generated response
            metadata: Additional context

        Returns:
            PolicyDecision with allowed/blocked status
        """
        # Evaluate using engine's output evaluation
        decision = await self.engine.evaluate_output(
            prompt=prompt,
            output=output,
            metadata=metadata
        )

        return decision

    def create_friendly_block_response(self, decision: PolicyDecision, is_streaming: bool = False) -> JSONResponse | StreamingResponse:
        """
        Create user-friendly OpenAI-compatible response for blocked requests.

        Instead of returning a 403 error, return a 200 OK with a friendly
        assistant message explaining the policy violation.

        Args:
            decision: PolicyDecision that resulted in BLOCK
            is_streaming: Whether to return a streaming response (SSE format)

        Returns:
            JSONResponse or StreamingResponse formatted as OpenAI chat completion
        """
        import time

        # Create user-friendly message about what was detected
        detected_types = []
        for violation in decision.violations:
            # Extract the type from the message (e.g., "Detected 1 instance(s) of ssn")
            message = violation.message.lower()
            if "ssn" in message:
                detected_types.append("Social Security Number")
            elif "email" in message:
                detected_types.append("email address")
            elif "credit" in message or "card" in message:
                detected_types.append("credit card number")
            elif "phone" in message:
                detected_types.append("phone number")
            elif "location" in message or "address" in message:
                detected_types.append("location information")
            else:
                detected_types.append("sensitive information")

        detected_str = ", ".join(detected_types) if detected_types else "sensitive information"

        friendly_message = (
            f"🔒 I'm sorry, but I cannot process your request because it contains {detected_str}. "
            f"For your security and privacy, IronClad-AI policies prevent me from handling sensitive personal information.\n\n"
            f"Please remove any sensitive data and try again. If you believe this is an error, "
            f"please contact your system administrator."
        )

        response_id = f"chatcmpl-ironclad-block-{int(time.time())}"
        created_time = int(time.time())

        if is_streaming:
            # Return streaming response in OpenAI Responses API SSE format
            async def generate_stream():
                msg_id = f"msg_{response_id}"

                # Event 1: response.created
                yield f"event: response.created\n"
                yield f"data: {json.dumps({'type': 'response.created', 'sequence_number': 0, 'response': {'id': response_id, 'object': 'response', 'created_at': created_time, 'status': 'in_progress'}})}\n\n"

                # Event 2: response.in_progress
                yield f"event: response.in_progress\n"
                yield f"data: {json.dumps({'type': 'response.in_progress', 'sequence_number': 1, 'response': {'id': response_id, 'object': 'response', 'created_at': created_time, 'status': 'in_progress'}})}\n\n"

                # Event 3: response.output_item.added
                yield f"event: response.output_item.added\n"
                yield f"data: {json.dumps({'type': 'response.output_item.added', 'sequence_number': 2, 'output_index': 0, 'item': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': []}})}\n\n"

                # Event 4: response.content_part.added
                yield f"event: response.content_part.added\n"
                yield f"data: {json.dumps({'type': 'response.content_part.added', 'sequence_number': 3, 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'text', 'text': ''}})}\n\n"

                # Event 5: response.output_text.delta (with content)
                yield f"event: response.output_text.delta\n"
                yield f"data: {json.dumps({'type': 'response.output_text.delta', 'sequence_number': 4, 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'delta': friendly_message})}\n\n"

                # Event 6: response.content_part.done
                yield f"event: response.content_part.done\n"
                yield f"data: {json.dumps({'type': 'response.content_part.done', 'sequence_number': 5, 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'text', 'text': friendly_message}})}\n\n"

                # Event 7: response.output_text.done
                yield f"event: response.output_text.done\n"
                yield f"data: {json.dumps({'type': 'response.output_text.done', 'sequence_number': 6, 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'text': friendly_message})}\n\n"

                # Event 8: response.output_item.done
                yield f"event: response.output_item.done\n"
                yield f"data: {json.dumps({'type': 'response.output_item.done', 'sequence_number': 7, 'output_index': 0, 'item': {'id': msg_id, 'type': 'message', 'status': 'completed', 'content': [{'type': 'output_text', 'text': friendly_message}], 'role': 'assistant'}})}\n\n"

                # Event 9: response.completed (KEY - not response.done!)
                yield f"event: response.completed\n"
                yield f"data: {json.dumps({'type': 'response.completed', 'sequence_number': 8, 'response': {'id': response_id, 'object': 'response', 'created_at': created_time, 'status': 'completed', 'status_details': None, 'output': [{'id': msg_id, 'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': friendly_message}]}], 'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'input_tokens_details': {'cached_tokens': 0, 'text_tokens': 0, 'audio_tokens': 0, 'image_tokens': 0, 'cached_tokens_details': {'text_tokens': 0, 'audio_tokens': 0, 'image_tokens': 0}}, 'output_tokens_details': {'text_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0}}}})}\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                status_code=200
            )
        else:
            # Return non-streaming JSON response
            response_body = {
                "id": response_id,
                "object": "chat.completion",
                "created": created_time,
                "model": "ironclad-policy-enforcer",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": friendly_message,
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

            return JSONResponse(
                status_code=200,  # Return 200 so Onyx displays it as a normal message
                content=response_body
            )

    def create_error_response(self, decision: PolicyDecision) -> JSONResponse:
        """
        Create HTTP 403 error response for blocked requests.

        DEPRECATED: Use create_friendly_block_response() instead for better UX.

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
                    "action": decision.action if isinstance(decision.action, str) else decision.action.value,
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
) -> tuple[bool, Optional[dict], Optional[JSONResponse | Any]]:
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
        # Return friendly user message (200 OK so it displays in chat)
        # Check if client requested streaming
        is_streaming = request_body.get("stream", False)
        friendly_response = enforcer.create_friendly_block_response(decision, is_streaming=is_streaming)
        return False, None, friendly_response

    elif decision.action == PolicyAction.REDACT:
        # Modify request body with redacted content
        modified_body = request_body.copy()

        # Replace content in messages (OpenAI format)
        if "messages" in modified_body and decision.modified_content:
            modified_body["messages"][-1]["content"] = decision.modified_content

        # Replace content in input (OpenAI Responses API and Onyx/custom format)
        if "input" in modified_body and decision.modified_content:
            input_data = modified_body["input"]

            # Handle simple string input (OpenAI Responses API format)
            if isinstance(input_data, str):
                modified_body["input"] = decision.modified_content
            # Handle array of messages (Onyx/custom format)
            elif isinstance(input_data, list):
                # Find the last user message in input array
                for i in range(len(input_data) - 1, -1, -1):
                    message = input_data[i]
                    if isinstance(message, dict) and "content" in message:
                        content = message["content"]
                        # Handle nested content structure
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    item["text"] = decision.modified_content
                                    break
                        else:
                            modified_body["input"][i]["content"] = decision.modified_content
                        break

        # Replace prompt (Anthropic format)
        if "prompt" in modified_body and decision.modified_content:
            modified_body["prompt"] = decision.modified_content

        return True, modified_body, None

    else:
        # ALLOW or WARN - proceed unchanged
        return True, None, None


async def enforce_output_policy(
    prompt: str,
    output: str,
    metadata: Optional[dict] = None,
    is_streaming: bool = False
) -> tuple[bool, Optional[str]]:
    """
    Enforce policy on LLM output.

    Args:
        prompt: Original user prompt
        output: LLM generated response
        metadata: Additional context
        is_streaming: Whether this is a streaming response

    Returns:
        Tuple of (allowed, replacement_output)
        - allowed: True if output should be returned to user
        - replacement_output: Modified output or block message if blocked
    """
    # Check if policies are enabled
    if os.getenv("IRONCLAD_ENABLED", "true").lower() != "true":
        return True, None

    # Get enforcer
    enforcer = get_policy_enforcer()

    # Evaluate output
    decision = await enforcer.evaluate_output(
        prompt=prompt,
        output=output,
        metadata=metadata
    )

    # Handle decision
    if decision.action == PolicyAction.BLOCK:
        # Return friendly block message
        friendly_response = enforcer.create_friendly_block_response(decision, is_streaming=is_streaming)

        # Extract content from the response
        if is_streaming:
            # For streaming, we can't easily use StreamingResponse here
            # Instead, return a simple message
            block_message = (
                "🔒 This response was blocked by IronClad security policies because it contains sensitive or unsafe content. "
                "The detected issues include: "
            )
            detected_scanners = []
            for violation in decision.violations:
                if hasattr(violation, 'scanner'):
                    detected_scanners.append(violation.scanner)
                elif hasattr(violation, 'policy_type'):
                    detected_scanners.append(violation.policy_type)

            if detected_scanners:
                block_message += ", ".join(set(detected_scanners))
            else:
                block_message += "security violations"

            block_message += ". Please contact your system administrator if you believe this is an error."

            return False, block_message
        else:
            # For non-streaming, extract the message content
            if isinstance(friendly_response, JSONResponse):
                import json
                # Parse the JSON body to get the content
                body = json.loads(friendly_response.body.decode())
                if "choices" in body and len(body["choices"]) > 0:
                    message_content = body["choices"][0].get("message", {}).get("content", "Response blocked by security policy")
                    return False, message_content

            # Fallback message
            return False, "🔒 Response blocked by IronClad security policy."

    elif decision.action == PolicyAction.REDACT:
        # Return modified output
        if decision.modified_content:
            return True, decision.modified_content
        else:
            # No modification available, allow original
            return True, None

    else:
        # ALLOW or WARN - proceed unchanged
        return True, None
