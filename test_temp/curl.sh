#!/bin/bash

curl http://localhost:50000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "秋意渐浓，风里藏着桂花的甜香，漫山枫叶如火，铺就一地斑斓。阳光透过稀疏的枝叶，洒下斑驳光影，温柔了岁月。落叶翩跹，似在低语季节的更迭，每一片都承载着时光的诗意。田野里稻穗低垂，金黄一片，是大地丰收的笑颜。秋，是沉淀，是静美，是岁月写给大地的温柔情书，让人心生宁静与眷恋。晨雾轻笼，湖面泛起微澜，倒映着岸边的红枫与黄栌，宛如一幅浓墨重彩的油画。偶有飞鸟掠过，划破宁静，又归于沉寂。秋虫在草丛间低吟，为这静谧添了几分生机。傍晚时分，夕阳将天边染成橘红，余晖洒在肩头，暖意融融，仿佛连时光都放慢了脚步。秋天不似春的喧闹，夏的炽热，冬的凛冽，它以独有的从容，将万物沉淀成一幅静美的画卷。每一缕风，都在诉说着岁月的故事。在这温柔的季节里，心也跟着沉静下来，仿佛能听见时光流淌的声音。请求二。"}],
    "max_tokens":3,
    "agent_hint": {
        "session_id": "sub-1",
        "parent_session_id": "main-0",
        "cache_control": {"type": "ephemeral", "ttl": 300, "msg_offset": 6},
        "context_management": {
          "manage_request":  false,
          "edits": [{"type": "offload", "start": 6, "end": 9, "target":"messages"}]
        }
    }
  }'