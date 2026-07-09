#!/bin/bash



curl http://localhost:50000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "",
    "messages": [{"role": "user", "content": ""}],
    "max_tokens":1,
    "agent_hint": {
        "session_id": "sub-1",
        "parent_session_id": "main-0",
        "cache_control": {"type": "ephemeral", "ttl": 10, "msg_offset": 6, "block_offset": 1, "token_offset": 2},
        "context_management": {
          "manage_request":  true,
          "edits": [{"type": "evict", "start": 6, "end": 9, "target":"messages", "block_start": 3, "block_end": 4},
            {"type": "offload", "start": 6, "end": 9, "target":"messages", "block_start": 3, "block_end": 4},
            {"type": "prefetch", "start": 6, "end": 9, "target":"messages", "block_start": 3, "block_end": 4}]
        }
    }
  }'
