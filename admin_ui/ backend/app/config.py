import yaml
import pathlib
import asyncio
import aiofiles

CONFIG_PATH = pathlib.Path("/app/config/config.yaml")   # mounted read‑only volume
RELOAD_ENDPOINT = "/api/v1/reload-config"             # Bot listens for this

async def load_config() -> dict:
    async with aiofiles.open(CONFIG_PATH, "r") as f:
        content = await f.read()
    return yaml.safe_load(content)


async def save_config(new_cfg: dict):
    # Write to a temporary file then atomically replace
    tmp_path = CONFIG_PATH.with_suffix(".tmp")
    async with aiofiles.open(tmp_path, "w") as f:
        await f.write(yaml.safe_dump(new_cfg, sort_keys=False))
    tmp_path.replace(CONFIG_PATH)

    # Notify the running bot (the bot watches the file via watchdog)
    # If you need an explicit HTTP trigger, uncomment:
    # async with aiohttp.ClientSession() as s:
    #     await s.post(f"http://citadel-bot:8000/api/v1/reload-config")

STAGING_PATH = Path("/app/config/config_staging.yaml")

async def propose_config(new_cfg: dict):
    # Write to staging area (still inside the container, writable)
    async with aiofiles.open(STAGING_PATH, "w") as f:
        await f.write(yaml.safe_dump(new_cfg, sort_keys=False))
    return {"msg": "Config proposal saved – create a PR to promote"}
