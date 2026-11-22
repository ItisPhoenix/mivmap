"""Evidence that the default suite is offline and isolated from private config."""
from __future__ import annotations

import asyncio
import socket
import threading
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from app.pipeline import embeddings as E
from conftest import NETWORK_BLOCKED


def test_configuration_is_environment_driven_not_hardcoded(monkeypatch):
    """Build a fresh Settings with a stubbed environment; never read the real .env."""
    from app import config

    env = {
        "EMBEDDING_BASE_URL": "https://example.invalid",
        "EMBEDDING_MODEL": "env-model",
        "EMBEDDING_API_KEY": "env-key",
        "EMBEDDING_LOCAL_MODEL": "env-local-model",
        "EMBEDDING_TIMEOUT": "12.5",
    }
    monkeypatch.setattr(config, "os", types.SimpleNamespace(
        getenv=lambda name, default=None: env.get(name, default)
    ))
    settings = config.Settings()

    assert settings.embedding_base_url == env["EMBEDDING_BASE_URL"]
    assert settings.embedding_model == env["EMBEDDING_MODEL"]
    assert settings.embedding_api_key == env["EMBEDDING_API_KEY"]
    assert settings.embedding_local_model == env["EMBEDDING_LOCAL_MODEL"]
    assert settings.embedding_timeout == 12.5
    assert settings.embedding_batch_size == 8  # documented default, no provider specifics


def test_ordinary_tests_cannot_use_private_credentials(monkeypatch):
    """The offline fixture replaces embedding settings; the .env values stay private."""
    from app import config

    assert E.settings is not config.settings
    assert E.settings.embedding_base_url == ""
    assert E.settings.embedding_model == ""
    assert E.settings.embedding_api_key == ""
    assert E.settings.embedding_local_model == ""

    def forbidden_remote_client(*_args, **_kwargs):
        raise AssertionError("remote client used by an ordinary test")

    monkeypatch.setattr(E, "_remote_client", forbidden_remote_client)
    from test_pipeline import synthetic_transcript
    from app.pipeline.builder import build_skeleton

    mindmap = build_skeleton({
        "videoId": "TESTVIDEO01", "title": "Quantum",
        "transcript": synthetic_transcript(), "mode": "deep",
        "useLLM": False, "allowWhisper": False,
    })
    assert mindmap["root"]["children"]


def test_guard_blocks_sync_async_and_socket_network():
    with pytest.raises(AssertionError, match=NETWORK_BLOCKED):
        httpx.Client().post("https://example.invalid", json={})
    async def async_request():
        return await httpx.AsyncClient().get("https://example.invalid")

    with pytest.raises(AssertionError, match=NETWORK_BLOCKED):
        asyncio.run(async_request())
    with pytest.raises(AssertionError, match=NETWORK_BLOCKED):
        socket.create_connection(("example.invalid", 80), timeout=0.1)


def test_guard_blocks_tcp_udp_dns_and_urllib():
    # TCP connect
    with pytest.raises(AssertionError, match=NETWORK_BLOCKED):
        socket.socket().connect(("example.invalid", 80))
    # UDP sendto
    with pytest.raises(AssertionError, match=NETWORK_BLOCKED):
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("example.invalid", 53))
    # DNS resolution
    with pytest.raises(AssertionError, match=NETWORK_BLOCKED):
        socket.getaddrinfo("example.invalid", 80)
    with pytest.raises(AssertionError, match=NETWORK_BLOCKED):
        socket.gethostbyname("example.invalid")
    # urllib
    with pytest.raises(AssertionError, match=NETWORK_BLOCKED):
        urllib.request.urlopen("https://example.invalid")


def test_guard_permits_loopback():
    # A refused connection proves the call passed the guard (no AssertionError).
    with pytest.raises(OSError) as exc:
        socket.create_connection(("127.0.0.1", 1), timeout=0.1)
    assert not isinstance(exc.value, AssertionError)
    assert socket.getaddrinfo("localhost", 80)  # loopback DNS permitted
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.sendto(b"", ("127.0.0.1", 1))  # loopback UDP permitted


@pytest.fixture
def loopback_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_guard_permits_real_loopback_httpx_sync(loopback_server):
    with httpx.Client() as client:
        assert client.get(loopback_server).status_code == 200


def test_guard_permits_real_loopback_httpx_async(loopback_server):
    async def request():
        async with httpx.AsyncClient() as client:
            return (await client.get(loopback_server)).status_code

    assert asyncio.run(request()) == 200


def test_guard_blocks_additional_dns_apis():
    with pytest.raises(AssertionError, match=NETWORK_BLOCKED):
        socket.gethostbyname_ex("example.invalid")
    with pytest.raises(AssertionError, match=NETWORK_BLOCKED):
        socket.gethostbyaddr("external.invalid")
    assert socket.gethostbyname_ex("localhost")  # loopback resolution permitted


def test_guard_permits_in_process_mock_transport():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="ok"))
    with httpx.Client(transport=transport) as client:
        assert client.get("https://example.invalid").status_code == 200


def test_cache_counters_start_clean():
    stats = E.embedding_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["writes"] == 0
    assert stats["invalidated"] == 0
    assert stats["memory_entries"] == 0


def test_use_llm_false_still_uses_embeddings(monkeypatch):
    from test_pipeline import synthetic_transcript
    from app.pipeline.builder import build_skeleton

    created = []
    original = E.Embedder

    class CountingEmbedder(original):
        def __init__(self, corpus):
            created.append(len(corpus))
            super().__init__(corpus)

    monkeypatch.setattr(E, "Embedder", CountingEmbedder)
    mindmap = build_skeleton({
        "videoId": "TESTVIDEO01", "title": "Quantum",
        "transcript": synthetic_transcript(), "mode": "deep",
        "useLLM": False, "allowWhisper": False,
    })
    assert created, "useLLM=False must not disable the embedding pipeline"
    assert mindmap["root"]["children"]
