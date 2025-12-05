# ==========================
# admin_ui/cron/Dockerfile
# ==========================
FROM alpine:3.20

# Install cron + any utilities you need (bash, curl, etc.)
RUN apk add --no-cache bash curl tzdata && \
    cp /usr/share/zoneinfo/UTC /etc/localtime && \
    echo "UTC" > /etc/timezone

# Copy your cron scripts (e.g. the daily fetch, backup, etc.)
COPY scripts/ /opt/cqt/scripts/
COPY cronjob /etc/cron.d/cqt-cron

# Give the cron file proper permissions and install it
RUN chmod 0644 /etc/cron.d/cqt-cron && \
    crontab /etc/cron.d/cqt-cron

# Run cron in foreground
CMD ["crond", "-f"]
