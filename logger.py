# Example – place in src/logger.py (import everywhere)
import logging, json, sys

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        # If you have extra fields (e.g., order_id) attached via `extra=...`,
        # they will appear automatically.
        return json.dumps(log_record)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
root = logging.getLogger()
root.setLevel(logging.INFO)
root.handlers = [handler]
