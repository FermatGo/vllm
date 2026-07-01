# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from typing import Annotated, Any, ClassVar, Literal
from vllm.entrypoints.openai.engine.protocol import (
    OpenAIBaseModel,
)

class SessionRequest(OpenAIBaseModel):
    session_id: str

# class SessionRequest:
#     session_id: str

