from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from prediction_market.adapters.kalshi import (
    KALSHI_WS_PATH,
    CredentialsUnavailableError,
    build_orderbook_subscription,
    kalshi_auth_headers,
    load_kalshi_credentials,
    parse_orderbook_frame,
)


@pytest.fixture
def private_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def test_kalshi_signature_is_rsa_pss_sha256(
    private_key: rsa.RSAPrivateKey,
) -> None:
    headers = kalshi_auth_headers(
        "GET",
        "/trade-api/ws/v2?ignored=true",
        1_234,
        "key-id",
        private_key,
    )

    assert headers["KALSHI-ACCESS-KEY"] == "key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1234"
    private_key.public_key().verify(
        base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"], validate=True),
        b"1234GET/trade-api/ws/v2",
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_kalshi_auth_rejects_non_rsa_key() -> None:
    with pytest.raises(TypeError, match="RSA"):
        kalshi_auth_headers("GET", KALSHI_WS_PATH, 1_234, "key-id", object())


def test_credentials_load_only_from_key_id_and_key_path_references(
    tmp_path: Path,
    private_key: rsa.RSAPrivateKey,
) -> None:
    key_path = tmp_path / "kalshi-private.key"
    key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    credentials = load_kalshi_credentials(
        {
            "KALSHI_API_KEY_ID": "key-id",
            "KALSHI_PRIVATE_KEY_PATH": str(key_path),
            "KALSHI_PRIVATE_KEY": "must-never-be-consumed",
        }
    )

    assert credentials.key_id == "key-id"
    assert credentials.private_key_path == key_path.resolve()
    assert credentials.private_key.private_numbers() == private_key.private_numbers()
    assert "PRIVATE KEY" not in repr(credentials)


@pytest.mark.parametrize(
    "environ,reason",
    [
        ({}, "KALSHI_API_KEY_ID"),
        ({"KALSHI_API_KEY_ID": "key-id"}, "KALSHI_PRIVATE_KEY_PATH"),
    ],
)
def test_missing_kalshi_credential_reference_fails_closed(
    environ: dict[str, str], reason: str
) -> None:
    with pytest.raises(CredentialsUnavailableError, match=reason):
        load_kalshi_credentials(environ)


def test_orderbook_subscription_uses_current_plural_ticker_field() -> None:
    subscription = json.loads(
        build_orderbook_subscription(["KXNBA-ONE", "KXNBA-TWO"])
    )

    assert subscription == {
        "id": 1,
        "cmd": "subscribe",
        "params": {
            "channels": ["orderbook_delta"],
            "market_tickers": ["KXNBA-ONE", "KXNBA-TWO"],
            "use_yes_price": True,
        },
    }
    assert "market_ticker" not in subscription["params"]


def test_orderbook_parser_requires_sid_seq_and_documented_type() -> None:
    parsed = parse_orderbook_frame(
        b'{"type":"orderbook_delta","sid":7,"seq":9,'
        b'"msg":{"market_ticker":"KXNBA-ONE"}}'
    )
    assert parsed["sid"] == 7
    assert parsed["seq"] == 9

    with pytest.raises(ValueError, match="seq"):
        parse_orderbook_frame(
            b'{"type":"orderbook_delta","sid":7,'
            b'"msg":{"market_ticker":"KXNBA-ONE"}}'
        )
