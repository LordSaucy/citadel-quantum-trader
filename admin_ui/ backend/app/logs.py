import docker
from fastapi.responses import StreamingResponse

docker_client = docker.from_env()


def _log_generator(container_name: str):
    try:
        container = docker_client.containers.get(container_name)
        for line in container.logs(stream=True, tail=200):
            yield line.decode(errors="replace")
    except docker.errors.NotFound:
        yield f"❌ Container {container_name} not found\n"


def stream_logs(container_name: str):
    return StreamingResponse(_log_generator(container_name), media_type="text/plain")
