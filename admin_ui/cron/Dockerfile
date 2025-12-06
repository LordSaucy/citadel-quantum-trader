# -----------------------------------------------------------------
#  admin_ui/cron/Dockerfile
# -----------------------------------------------------------------
FROM alpine:3.20

# ---- Install runtime dependencies (example) --------------------
# (Add only the packages you really need – keep the image small)
RUN apk add --no-cache \
        python3~=3.12 \
        py3-pip \
        bash \
        tzdata && \
    # Clean up any apk cache (already done by --no-cache)

# ---- Create a non‑root user ------------------------------------
# uid/gid 1000 is the conventional “regular user” ID on many systems.
# You can change these numbers if your organization reserves a different range.
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=cqtcron

RUN addgroup -g ${GROUP_ID} ${USER_NAME} && \
    adduser -D -u ${USER_ID} -G ${USER_NAME} -s /bin/sh ${USER_NAME}

# ---- Copy application code ---------------------------------------
WORKDIR /app
COPY . /app

# ---- Adjust ownership of copied files ---------------------------
# (Ensures the non‑root user can read/write where needed)
RUN chown -R ${USER_NAME}:${USER_NAME} /app

# ---- Switch to the non‑root user -------------------------------
USER ${USER_NAME}

# ---- Set a sane PATH (optional) --------------------------------
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# ---- Entrypoint / command ---------------------------------------
# Replace with whatever you normally run (e.g. a cron wrapper script)
CMD ["python3", "run_cron_jobs.py"]
