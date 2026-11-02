"""Tamper-proof tokens for the paused graph state.

Human-in-the-loop here is stateless: when the graph pauses, the server has nothing stored, so the paused state
travels to the browser and back. If the server trusted the state the browser sends back, a user could edit it
(e.g. add a bullet that never went through the honesty check) and the graph would continue from the edited state.

So the server sends the state inside a signed token: base64(zlib(json)) + "." + HMAC-SHA256(body, secret key).
On resume it recomputes the HMAC. A token that was changed, or signed with another key, is rejected.
The token is signed, not encrypted: whoever holds it can read the state (it's their own data) but can't change it.
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import zlib

# Fallback for local runs and tests only. It is public (it's in this file), so it gives no protection:
# set STATE_SIGNING_KEY in production (e.g. `python -c "import secrets; print(secrets.token_hex(32))"`).
DEV_KEY = "dev-only-signing-key-set-STATE_SIGNING_KEY"

log = logging.getLogger(__name__)


class InvalidToken(ValueError):
    """The token was modified, truncated, or signed with a different key."""


def signing_configured() -> bool:
    return bool(os.getenv("STATE_SIGNING_KEY"))


_warned = False


def _key() -> bytes:
    global _warned
    if not signing_configured() and not _warned:
        _warned = True
        log.warning("STATE_SIGNING_KEY is not set; using the public dev key (paused state is NOT tamper-proof)")
    return (os.getenv("STATE_SIGNING_KEY") or DEV_KEY).encode()


def _sign(body: str) -> str:
    return hmac.new(_key(), body.encode(), hashlib.sha256).hexdigest()


def seal(payload: dict) -> str:
    raw = zlib.compress(json.dumps(payload, ensure_ascii=False).encode())
    body = base64.urlsafe_b64encode(raw).decode()
    return f"{body}.{_sign(body)}"


def unseal(token: str) -> dict:
    body, _, sig = (token or "").rpartition(".")
    # compare_digest takes the same time whether the first or last character differs (no timing leak).
    # The signature is checked BEFORE decompressing, so untrusted bytes are never parsed.
    if not body or not hmac.compare_digest(sig, _sign(body)):
        raise InvalidToken("This paused run is no longer valid (it was modified, or the server key changed). Please run the agent again.")
    return json.loads(zlib.decompress(base64.urlsafe_b64decode(body)))
