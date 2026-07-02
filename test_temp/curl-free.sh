#!/bin/bash


curl http://localhost:50000/v1/sessions/session_id_1122/free \
  -H "Content-Type: application/json" \
  -d '{
    "model": "",
    "messages": [{"role": "user", "content": ""}],
    "max_tokens":1,
    "agent_hint": {
        "session_id": "main-0",
        "cache_control": {"type": "ephemeral", "ttl": 300, "msg_offset": 6},
        "context_management": {
          "manage_request":  true,
          "edits": [{"type": "offload", "start": 6, "end": 9, "target":"messages"}]
        }
    }
  }'
