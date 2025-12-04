@echo off
REM -------------------------------------------------
REM monitor_bot.bat – graceful restart wrapper
REM -------------------------------------------------
set CONTAINER_NAME=citadel-bot

rem 1️⃣ Ask Docker to stop the container (Docker will send SIGTERM)
docker stop %CONTAINER_NAME%

rem 2️⃣ Wait for it to finish (Docker respects stop_grace_period)
docker wait %CONTAINER_NAME%

rem 3️⃣ Start it again
docker start %CONTAINER_NAME%

echo Bot restarted gracefully.
