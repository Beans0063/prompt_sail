from typing import Annotated

import httpx
import json
from _datetime import datetime, timezone
from app.dependencies import get_logger, get_provider_pricelist, get_transaction_context
from app.policy_middleware import enforce_policy, enforce_output_policy
from fastapi import Depends, Request
from fastapi.responses import StreamingResponse
from lato import Application, TransactionContext
from projects.use_cases import get_project_by_slug
from raw_transactions.use_cases import store_raw_transactions
from starlette.background import BackgroundTask
from transactions.use_cases import store_transaction
from utils import ApiURLBuilder

from .app import app
import time
import uuid


def transform_chat_completions_to_responses_api(chunk: bytes, response_id: str = None, created_at: int = None) -> bytes:
    """
    Transform OpenAI Chat Completions streaming format to Responses API format.

    Converts:
        data: {"object":"chat.completion.chunk","delta":{"content":"..."}}
    To:
        event: response.output_text.delta
        data: {"index":0,"delta":"..."}

    Args:
        chunk: Raw chunk from Chat Completions API
        response_id: Response ID for Responses API events
        created_at: Unix timestamp for response creation

    Returns:
        Transformed chunk in Responses API format, or original chunk if not transformable
    """
    try:
        decoded = chunk.decode('utf-8')

        # Handle [DONE] marker
        if decoded.strip() == "data: [DONE]":
            return b"event: done\ndata: [DONE]\n\n"

        # Parse SSE format
        if not decoded.startswith('data: '):
            return chunk  # Not an SSE data line, pass through

        json_str = decoded[6:].strip()  # Remove "data: " prefix
        if not json_str or json_str == "[DONE]":
            return chunk

        data = json.loads(json_str)

        # Check if it's a Chat Completions chunk
        if data.get('object') != 'chat.completion.chunk':
            return chunk  # Not a chat completion, pass through

        # Extract delta content
        choices = data.get('choices', [])
        if not choices:
            return chunk

        choice = choices[0]
        delta = choice.get('delta', {})
        finish_reason = choice.get('finish_reason')

        # Generate IDs if not provided
        if not response_id:
            response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        if not created_at:
            created_at = int(time.time())

        # Transform based on delta content
        if finish_reason == 'stop':
            # Final chunk - send completion event
            completion_event = {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "model": data.get('model', 'gpt-3.5-turbo'),
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": []  # Would need to accumulate content
                }]
            }
            return f"event: response.completed\ndata: {json.dumps(completion_event)}\n\n".encode('utf-8')

        elif 'content' in delta and delta['content']:
            # Content delta
            delta_event = {
                "index": 0,
                "delta": delta['content']
            }
            return f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n".encode('utf-8')

        elif 'role' in delta:
            # Initial role assignment - send created event
            created_event = {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "model": data.get('model', 'gpt-3.5-turbo'),
                "status": "in_progress"
            }
            return f"event: response.created\ndata: {json.dumps(created_event)}\n\n".encode('utf-8')

        # Unknown delta type, pass through
        return chunk

    except Exception:
        # If transformation fails, pass through original chunk
        return chunk


def extract_llm_output_from_stream(stream_text: str, logger) -> str:
    """
    Extract LLM output text from OpenAI streaming response.

    Parses SSE (Server-Sent Events) format to extract the assistant's message.

    Args:
        stream_text: Full streaming response text
        logger: Logger instance

    Returns:
        Extracted text content or empty string
    """
    try:
        # OpenAI Responses API format: event lines like "event: response.output_text.delta"
        # followed by "data: {json with delta field}"
        output_parts = []

        for line in stream_text.split('\n'):
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    # Look for delta content in various fields
                    if 'delta' in data and isinstance(data['delta'], str):
                        output_parts.append(data['delta'])
                    elif 'delta' in data and isinstance(data['delta'], dict):
                        if 'text' in data['delta']:
                            output_parts.append(data['delta']['text'])
                        elif 'content' in data['delta']:
                            output_parts.append(data['delta']['content'])
                    # Also check direct text field
                    elif 'text' in data:
                        output_parts.append(data['text'])
                    # Check for completion format
                    elif 'choices' in data:
                        for choice in data['choices']:
                            if 'delta' in choice and 'content' in choice['delta']:
                                content = choice['delta']['content']
                                if content:
                                    output_parts.append(content)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Could not parse streaming line: {line[:100]}")
                    continue

        result = ''.join(output_parts)
        if result:
            logger.info(f"Extracted {len(result)} chars from {len(output_parts)} delta events")
        return result

    except Exception as e:
        logger.warning(f"Failed to extract LLM output from stream: {e}")
        return ""


def create_block_message_stream(block_message: str):
    """
    Create SSE stream events for a blocked message.

    Args:
        block_message: The block message to stream

    Returns:
        Generator of SSE event strings
    """
    import time

    response_id = f"ironclad-block-{int(time.time())}"
    msg_id = f"msg_{response_id}"
    created_time = int(time.time())

    # Minimal stream to deliver block message
    yield f"event: response.created\n"
    yield f"data: {json.dumps({'type': 'response.created', 'sequence_number': 0, 'response': {'id': response_id, 'object': 'response', 'created_at': created_time, 'status': 'in_progress'}})}\n\n"

    yield f"event: response.output_item.added\n"
    yield f"data: {json.dumps({'type': 'response.output_item.added', 'sequence_number': 1, 'output_index': 0, 'item': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': []}})}\n\n"

    yield f"event: response.content_part.added\n"
    yield f"data: {json.dumps({'type': 'response.content_part.added', 'sequence_number': 2, 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'text', 'text': ''}})}\n\n"

    yield f"event: response.output_text.delta\n"
    yield f"data: {json.dumps({'type': 'response.output_text.delta', 'sequence_number': 3, 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'delta': block_message})}\n\n"

    yield f"event: response.content_part.done\n"
    yield f"data: {json.dumps({'type': 'response.content_part.done', 'sequence_number': 4, 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'text', 'text': block_message}})}\n\n"

    yield f"event: response.completed\n"
    yield f"data: {json.dumps({'type': 'response.completed', 'sequence_number': 5, 'response': {'id': response_id, 'object': 'response', 'created_at': created_time, 'status': 'completed', 'output': [{'id': msg_id, 'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': block_message}]}]}})}\n\n"


async def iterate_stream(response, buffer):
    """
    Asynchronously iterate over the raw stream of a response and accumulate chunks in a buffer.

    This function asynchronously iterate over the raw stream of a response and accumulate chunks in a buffer
    for later processing.

    Parameters:
    - **response**: The response object containing the stream
    - **buffer**: List to store the accumulated response chunks

    Yields:
    - Chunks of the response data as they are received
    """
    async for chunk in response.aiter_raw():
        buffer.append(chunk)
        yield chunk


async def close_stream(
    app: Application,
    project_id,
    ai_provider_request,
    ai_provider_response,
    buffer,
    tags,
    ai_model_version,
    pricelist,
    request_time,
):
    """
    Process and store transaction data after stream completion.

    This function handles the post-streaming tasks, including storing transaction details
    and raw request/response data in the database.

    Parameters:
    - **app**: The Application instance
    - **project_id**: The unique identifier of the project
    - **ai_provider_request**: The original request object
    - **ai_provider_response**: The response object from the AI provider
    - **buffer**: Buffer containing the accumulated response data
    - **tags**: List of tags associated with the transaction
    - **ai_model_version**: Specific model version tag for cost calculation
    - **pricelist**: List of provider prices for cost calculation
    - **request_time**: Timestamp when the request was initiated
    """
    await ai_provider_response.aclose()
    with app.transaction_context() as ctx:
        data = ctx.call(
            store_transaction,
            project_id=project_id,
            ai_provider_request=ai_provider_request,
            ai_provider_response=ai_provider_response,
            buffer=buffer,
            tags=tags,
            ai_model_version=ai_model_version,
            pricelist=pricelist,
            request_time=request_time,
        )
        ctx.call(
            store_raw_transactions,
            request=ai_provider_request,
            request_content=data["request_content"],
            response=ai_provider_response,
            response_content=data["response_content"],
            transaction_id=data["transaction_id"],
        )


@app.api_route(
    "/api/{project_slug}/{provider_slug}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def reverse_proxy(
    project_slug: str,
    provider_slug: str,
    path: str,
    request: Request,
    ctx: Annotated[TransactionContext, Depends(get_transaction_context)],
    tags: str | None = None,
    ai_model_version: str | None = None,
    target_path: str | None = None,
):
    """
    Forward requests to AI providers and handle responses.

    This endpoint acts as a reverse proxy, forwarding requests to various AI providers
    while monitoring and storing transaction details. It handles streaming responses,
    calculates costs, and maintains transaction history.

    Parameters:
    - **project_slug**: The unique slug identifier of the project
    - **provider_slug**: The slug identifier of the AI provider
    - **path**: The API endpoint path to forward to
    - **request**: The incoming request object
    - **ctx**: The transaction context dependency
    - **tags**: Optional comma-separated list of tags for the transaction
    - **ai_model_version**: Optional specific model version for accurate cost calculation
    - **target_path**: Optional override for the target API path

    Returns:
    - A StreamingResponse object containing the provider's response

    Notes:
    - Automatically handles request/response streaming
    - Stores transaction details and raw data in the background
    - Calculates costs based on the provider's pricing
    - Supports various HTTP methods (GET, POST, PUT, PATCH, DELETE)
    """
    logger = get_logger(request)

    # if not request.state.is_handled_by_proxy:
    #     return RedirectResponse("/ui")
    # project = ctx.call(get_project_by_slug, slug=request.state.slug)

    tags = tags.split(",") if tags is not None else []
    project = ctx.call(get_project_by_slug, slug=project_slug)
    url = ApiURLBuilder.build(project, provider_slug, path, target_path)

    # todo: remove this, this logic should be in the use case
    pricelist = get_provider_pricelist(request)

    logger.debug(f"got projects for {project}")

    # Get the body as bytes for non-GET requests
    body = await request.body() if request.method != "GET" else None

    # ===== IRONCLAD POLICY ENFORCEMENT (NEW) =====
    # Parse request body for policy evaluation
    request_body = {}
    if body:
        try:
            request_body = json.loads(body.decode('utf-8'))
            logger.info(f"IronClad: Full request body keys: {list(request_body.keys())}")
            logger.info(f"IronClad: Request body: {json.dumps(request_body)[:500]}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Failed to parse request body for policy enforcement")
            request_body = {}

    # Enforce IronClad security policies
    allowed, modified_body, error_response = await enforce_policy(
        request=request,
        request_body=request_body,
        project_slug=project_slug,
        provider_slug=provider_slug
    )

    logger.info(f"IronClad: Policy decision - allowed={allowed}, modified={modified_body is not None}")

    if not allowed:
        # Request blocked by policy - return 403 error
        logger.warning(f"Request blocked by IronClad policy: {project_slug}/{provider_slug}")
        return error_response

    if modified_body:
        # Content was redacted - use modified body
        logger.info(f"Request content redacted by IronClad policy")
        body = json.dumps(modified_body).encode('utf-8')
    # ===== END IRONCLAD POLICY ENFORCEMENT =====

    # Make the request to the upstream server
    client = httpx.AsyncClient()
    # todo: copy timeout from request, temporary set to 100s
    timeout = httpx.Timeout(100.0, connect=50.0)

    # Exclude content-length when body was modified (httpx will recalculate)
    excluded_headers = ("host", "content-length") if modified_body else ("host",)

    request_time = datetime.now(tz=timezone.utc)
    ai_provider_request = client.build_request(
        method=request.method,
        url=url,
        headers={
            k: v for k, v in request.headers.items() if k.lower() not in excluded_headers
        },
        params=request.query_params,
        content=body,
        timeout=timeout,
    )
    logger.debug(f"Requesting on: {url}")
    ai_provider_response = await client.send(ai_provider_request, stream=True, follow_redirects=True)

    buffer = []

    # Check if output scanning is enabled
    import os
    output_scanning_enabled = os.getenv("IRONCLAD_OUTPUT_SCANNING", "false").lower() == "true"

    # Only use buffering+scanning if output scanners are enabled
    if output_scanning_enabled:
        # Add logging for successful streaming responses + output scanning
        async def iterate_stream_with_scanning(response, buffer, prompt, metadata, transform=False):
            """Stream response chunks and apply output scanning."""
            chunk_count = 0
            collected_chunks = []

            # Generate consistent IDs for transformation if needed
            response_id = None
            created_at = None
            if transform:
                response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                created_at = int(time.time())

            # Collect all chunks first
            async for chunk in response.aiter_raw():
                buffer.append(chunk)
                collected_chunks.append(chunk)
                chunk_count += 1
                # Log chunks to understand format
                try:
                    decoded = chunk.decode('utf-8')
                    logger.info(f"STREAM CHUNK {chunk_count}: {decoded[:300]}")
                except:
                    logger.info(f"STREAM CHUNK {chunk_count}: [binary data]")

            # Try to extract LLM output text from collected chunks
            full_text = b''.join(collected_chunks).decode('utf-8', errors='ignore')
            llm_output = extract_llm_output_from_stream(full_text, logger)

            # If we extracted output, scan it
            if llm_output:
                logger.info(f"IronClad OUTPUT SCAN: Extracted {len(llm_output)} chars from response")

                # Enforce output policy
                allowed, replacement = await enforce_output_policy(
                    prompt=prompt,
                    output=llm_output,
                    metadata=metadata,
                    is_streaming=True
                )

                if not allowed:
                    # Response blocked - replace with block message
                    logger.warning(f"IronClad: LLM output blocked by policy")

                    # Create a new streaming response with block message
                    block_event = create_block_message_stream(replacement)
                    for event_chunk in block_event:
                        yield event_chunk.encode('utf-8')
                    return

            # Output allowed or no output detected - yield chunks (with transformation if needed)
            for chunk in collected_chunks:
                if transform:
                    # Transform Chat Completions format to Responses API format
                    transformed_chunk = transform_chat_completions_to_responses_api(
                        chunk, response_id, created_at
                    )
                    yield transformed_chunk
                else:
                    yield chunk

        # Extract user prompt for output scanning
        user_prompt = ""
        if request_body and "messages" in request_body:
            messages = request_body.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_prompt = str(msg.get("content", ""))
                    break

        # Build metadata for output scanning
        output_metadata = {
            "project": project_slug,
            "provider": provider_slug,
            "ip_address": request.client.host if request.client else "unknown",
        }

        return StreamingResponse(
            iterate_stream_with_scanning(
                ai_provider_response,
                buffer,
                user_prompt,
                output_metadata,
                transform=needs_transformation
            ),
            status_code=ai_provider_response.status_code,
            headers=ai_provider_response.headers,
            background=BackgroundTask(
                close_stream,
                ctx["app"],
                project.id,
                ai_provider_request,
                ai_provider_response,
                buffer,
                tags,
                ai_model_version,
                pricelist,
                request_time,
            ),
        )
    else:
        # Output scanning disabled - use normal streaming (no buffering)
        logger.debug("Output scanning disabled - using real-time streaming")

        # Detect if we need to transform Chat Completions to Responses API format
        needs_transformation = '/responses' in path if path else False
        logger.debug(f"Responses API transformation: {'enabled' if needs_transformation else 'disabled'} (path: {path})")

        async def iterate_stream_passthrough(response, buffer, transform=False):
            """Stream response chunks in real-time without buffering."""
            response_id = None
            created_at = None

            if transform:
                # Generate consistent IDs for the entire response
                response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                created_at = int(time.time())

            async for chunk in response.aiter_raw():
                buffer.append(chunk)

                if transform:
                    # Transform Chat Completions format to Responses API format
                    transformed_chunk = transform_chat_completions_to_responses_api(
                        chunk, response_id, created_at
                    )
                    yield transformed_chunk
                else:
                    yield chunk

        return StreamingResponse(
            iterate_stream_passthrough(ai_provider_response, buffer, needs_transformation),
            status_code=ai_provider_response.status_code,
            headers=ai_provider_response.headers,
            background=BackgroundTask(
                close_stream,
                ctx["app"],
                project.id,
                ai_provider_request,
                ai_provider_response,
                buffer,
                tags,
                ai_model_version,
                pricelist,
                request_time,
            ),
        )
