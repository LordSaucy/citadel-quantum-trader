# -----------------------------------------------------------------
# cqt_env.sh – environment variables used by go_live.sh
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# SSH / DigitalOcean
# -----------------------------------------------------------------
SSH_USER="root"                     # user that can run Docker on the droplets
SSH_KEY_PATH="$HOME/.ssh/id_rsa"    # private key used for SSH (must be added to the droplets)

# -----------------------------------------------------------------
# Droplet IPs (private VPC addresses)
# -----------------------------------------------------------------
PRIMARY_IP="10.10.0.10"   # cqt-engine-primary.private IP
STANDBY_IP="10.10.0.11"   # cqt-engine-standby.private IP
DB_IP="10.10.0.12"        # cqt-db.private IP (used by validation only)
MONITOR_IP="10.10.0.13"   # cqt-monitor.private IP (Prometheus)
GRAFANA_IP="10.10.0.14"   # cqt-grafana.private IP
BACKUP_IP="10.10.0.15"    # cqt-backup.private IP (optional)

# -----------------------------------------------------------------
# Container registry & image name
# -----------------------------------------------------------------
REGISTRY="ghcr.io"
REPO_OWNER="your-org-or-username"
IMAGE_NAME="cqt-engine"
# The CI pipeline tags the image with the git SHA; we also keep a `latest` tag.
FULL_IMAGE="${REGISTRY}/${REPO_OWNER}/${IMAGE_NAME}:latest"

# -----------------------------------------------------------------
# Load Balancer (optional – update backend list after restart)
# -----------------------------------------------------------------
# If you use the DigitalOcean Managed Load Balancer, set its ID here.
# Leave empty if you don’t want the script to touch the LB.
LB_ID="your-load-balancer-id"

# -----------------------------------------------------------------
# Misc
# -----------------------------------------------------------------
DEPLOY_ROOT="/opt/cqt"          # absolute path where the repo lives on each droplet
LOG_FILE="/var/log/cqt_go_live_$(date +%Y%m%d_%H%M%S).log"
# -----------------------------------------------------------------
