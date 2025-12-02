import redis

r = redis.Redis(host="redis", port=6379, db=0)   # you already have a Redis container

def set_bucket_flag(bucket_id: int, enabled: bool):
    r.set(f"bucket:{bucket_id}:use_schedule", int(enabled))

def get_bucket_flag(bucket_id: int) -> bool:
    val = r.get(f"bucket:{bucket_id}:use_schedule")
    return bool(int(val)) if val is not None else True   # default = True
