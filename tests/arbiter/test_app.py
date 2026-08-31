from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from freetoken.arbiter.app import ArbiterConfig, build_arbiter_app
from freetoken.arbiter.backends import ActiveBackend
from freetoken.arbiter.model import ModelId


class FakeController:
    def __init__(self) -> None:
        self.calls: list[ModelId] = []
        self.releases: list[ModelId] = []

    async def prepare(self, model_id: ModelId) -> ActiveBackend:
        self.calls.append(model_id)
        return ActiveBackend(model_id, "http://backend.test", "backend-model", "fake")

    async def release(self, model_id: ModelId) -> None:
        self.releases.append(model_id)


def _upstream_handler(seen: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"id": "cmpl-1", "object": "chat.completion", "model": body["model"], "choices": []},
        )

    return handler


def _client(controller: FakeController, seen: list[dict]) -> TestClient:
    transport = httpx.MockTransport(_upstream_handler(seen))
    upstream = httpx.AsyncClient(transport=transport)
    app = build_arbiter_app(ArbiterConfig(queue_timeout_s=1.0), controller, upstream)
    return TestClient(app)


def test_models_always_lists_both_public_ids():
    controller = FakeController()
    with _client(controller, []) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == ["ornith-35b", "gemma-4-e2b"]


def test_unknown_model_is_404_without_backend_transition():
    controller = FakeController()
    with _client(controller, []) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "unknown", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 404
    assert controller.calls == []


def test_request_model_is_rewritten_and_lease_is_released_after_buffered_response():
    controller = FakeController()
    seen: list[dict] = []
    with _client(controller, seen) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ornith-35b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert controller.calls == [ModelId.ORNITH]
    assert controller.releases == [ModelId.ORNITH]
    assert seen == [{
        "model": "backend-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }]


def test_streaming_response_releases_lease_after_sse_body_is_consumed():
    controller = FakeController()
    seen: list[dict] = []

    def streaming_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"
            ),
        )

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(streaming_handler))
    app = build_arbiter_app(ArbiterConfig(queue_timeout_s=1.0), controller, upstream)
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gemma-4-e2b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as response:
            chunks = list(response.iter_bytes())

    assert response.status_code == 200
    assert b"[DONE]" in b"".join(chunks)
    assert seen[0]["model"] == "backend-model"
    assert controller.releases == [ModelId.GEMMA]
