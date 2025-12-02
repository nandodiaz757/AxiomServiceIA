import json
import redis.asyncio as redis
import numpy as np
from datetime import datetime

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

HYBRID_STREAM = "train_hybrid"
GENERAL_STREAM = "train_general"

# -----------------------------
# JSON Sanitizer
# -----------------------------
def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)  # fallback seguro


# -----------------------------
# SEND MESSAGE (Seguro)
# -----------------------------
async def send_message(queue_name, data):
    safe_json = json.dumps(data, default=make_json_safe, ensure_ascii=False)
    await redis_client.xadd(queue_name, {"data": safe_json})


# -----------------------------
# RECEIVE MESSAGES
# -----------------------------
async def receive_messages(queue_name, count=10):
    msgs = await redis_client.xread({queue_name: "0-0"}, count=count, block=0)
    if not msgs:
        return []
    result = []
    for _, items in msgs:
        for msg_id, fields in items:
            result.append({
                "id": msg_id,
                "body": json.loads(fields["data"])
            })
    return result


# -----------------------------
# DELETE MESSAGE
# -----------------------------
async def delete_message(queue_name, msg_id):
    await redis_client.xdel(queue_name, msg_id)