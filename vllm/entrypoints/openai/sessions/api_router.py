# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.utils import random_uuid
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.sessions.protocol import SessionRequest
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def batch_chat(request: Request) -> OpenAIServingChatBatch | None:
    return request.app.state.openai_serving_chat_batch


@router.post(
    "/v1/sessions/{session_id}/free",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def free_session(session_id: str, request: ChatCompletionRequest, raw_request: Request):

    logger.warning(f'===== free_session, session_id = {session_id}, request = {request}, raw_request = {raw_request}')

    # # state.engine_client = engine_client
    # #     state.log_stats = not args.disable_log_stats
    # #     state.vllm_config = vllm_config
    # #     state.args = args
    # # ===== engine_client = <vllm.v1.engine.async_llm.AsyncLLM object at 0xfffbadc42310>
    # engine_client = raw_request.app.state.engine_client
    # await engine_client.free_session(session_id)
    # logger.warning(f'===== engine_client = {engine_client}')
    #
    # return {
    #     "session_id": "sub-1",
    #     "freed_blocks": 12,
    #     "orphaned_blocks": 8,
    #     "children_freed": [
    #         {"session_id": "sub-1-child", "freed_blocks": 5, "orphaned_blocks": 3}
    #     ]
    # }

    request.session_management_flag = 1

    logger.warning(f'===== free_session, request = {request}')

    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")




def attach_router(app: FastAPI):
    app.include_router(router)
