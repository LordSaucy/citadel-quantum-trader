touch /opt/config/use_optimised_cfg.flag
# Signal the bot to reload immediately (the watcher already polls every 5 s)
echo "reload" > /opt/config/reload_now   # optional if you have a manual‑reload trigger
