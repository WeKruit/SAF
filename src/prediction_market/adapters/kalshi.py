"""Kalshi authentication and authenticated orderbook wire protocol."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from prediction_market.adapters.base import ProtocolError


KALSHI_WS_PATH = "/trade-api/ws/v2"
KALSHI_WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
KALSHI_DEMO_WS_URL = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"


class CredentialsUnavailableError(RuntimeError):
    """Kalshi recording is blocked until key references are provisioned."""


@dataclass(frozen=True, slots=True)
class KalshiCredentials:
    key_id: str
    private_key_path: Path
    private_key: rsa.RSAPrivateKey = field(repr=False)


def _required_reference(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if type(value) is not str or not value.strip():
        raise CredentialsUnavailableError(f"missing environment reference {name}")
    return value


def load_kalshi_credentials(
    environ: Mapping[str, str] | None = None,
) -> KalshiCredentials:
    """Load a key ID and PEM path reference without accepting inline key material."""

    source = os.environ if environ is None else environ
    key_id = _required_reference(source, "KALSHI_API_KEY_ID")
    path_text = _required_reference(source, "KALSHI_PRIVATE_KEY_PATH")
    try:
        key_path = Path(path_text).expanduser().resolve(strict=True)
        if not key_path.is_file():
            raise OSError("not a regular file")
        material = key_path.read_bytes()
        loaded = serialization.load_pem_private_key(material, password=None)
    except (OSError, TypeError, ValueError) as exc:
        raise CredentialsUnavailableError(
            "KALSHI_PRIVATE_KEY_PATH does not reference a readable unencrypted PEM key"
        ) from exc
    if not isinstance(loaded, rsa.RSAPrivateKey):
        raise CredentialsUnavailableError(
            "KALSHI_PRIVATE_KEY_PATH must reference an RSA private key"
        )
    if loaded.key_size < 2048:
        raise CredentialsUnavailableError("Kalshi RSA private key must be at least 2048 bits")
    return KalshiCredentials(
        key_id=key_id,
        private_key_path=key_path,
        private_key=loaded,
    )


def kalshi_auth_headers(
    method: str,
    path: str,
    timestamp_ms: int,
    key_id: str,
    private_key: rsa.RSAPrivateKey,
) -> dict[str, str]:
    """Sign timestamp + method + query-free path with RSA-PSS/SHA-256."""

    if type(method) is not str or not method:
        raise ValueError("method must be non-empty")
    if type(path) is not str or not path.startswith("/") or "://" in path:
        raise ValueError("path must be an absolute URL path")
    if type(timestamp_ms) is not int or timestamp_ms < 0:
        raise ValueError("timestamp_ms must be a non-negative integer")
    if type(key_id) is not str or not key_id.strip():
        raise ValueError("key_id must be non-empty")
    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise TypeError("private_key must be an RSA private key")

    signed_path = path.split("?", 1)[0]
    timestamp = str(timestamp_ms)
    message = f"{timestamp}{method.upper()}{signed_path}".encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }


def _unique_tickers(market_tickers: Sequence[str]) -> tuple[str, ...]:
    if isinstance(market_tickers, (str, bytes)):
        raise ValueError("market_tickers must be a sequence")
    tickers = tuple(market_tickers)
    if not tickers:
        raise ValueError("market_tickers must not be empty")
    if any(type(value) is not str or not value or value.strip() != value for value in tickers):
        raise ValueError("market_tickers must contain non-empty strings")
    if len(set(tickers)) != len(tickers):
        raise ValueError("market_tickers must not contain duplicates")
    return tickers


def build_orderbook_subscription(market_tickers: Sequence[str]) -> str:
    """Use explicit yes-price semantics and the current plural ticker field."""

    tickers = _unique_tickers(market_tickers)
    return json.dumps(
        {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": list(tickers),
                "use_yes_price": True,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_orderbook_frame(payload: bytes) -> dict[str, Any]:
    """Parse one documented snapshot/delta after exact raw persistence."""

    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Kalshi frame is not valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ProtocolError("Kalshi orderbook frame must be an object")
    if value.get("type") not in {"orderbook_snapshot", "orderbook_delta"}:
        raise ProtocolError("Kalshi orderbook frame type is invalid")
    if type(value.get("sid")) is not int:
        raise ProtocolError("Kalshi orderbook sid is required")
    if type(value.get("seq")) is not int:
        raise ProtocolError("Kalshi orderbook seq is required")
    if type(value.get("msg")) is not dict:
        raise ProtocolError("Kalshi orderbook msg is required")
    return value


__all__ = [
    "CredentialsUnavailableError",
    "KALSHI_DEMO_WS_URL",
    "KALSHI_WS_PATH",
    "KALSHI_WS_URL",
    "KalshiCredentials",
    "build_orderbook_subscription",
    "kalshi_auth_headers",
    "load_kalshi_credentials",
    "parse_orderbook_frame",
]
