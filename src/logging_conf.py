"""Single place to configure logging so every module logs consistently."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

# Anything matching these substrings is scrubbed from log records so a token
# never lands in a hosted platform's log stream.
_SECRET_KEYS = ("token", "api_key", "apikey", "authorization", "secret")


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = str(record.getMessage())
        except Exception:
            return True
        lowered = msg.lower()
        if any(k in lowered for k in _SECRET_KEYS):
            # Redact anything that looks like a long opaque credential.
            import re

            record.msg = re.sub(r"[A-Za-z0-9_\-\.]{24,}", "<redacted>", msg)
            record.args = ()
        return True


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
    )
    handler.addFilter(_RedactFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    # httpx is chatty at INFO and logs full URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
