from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from benchmarks.gemma_speaker_memory_acceptance import VOICE_SYSTEM_PROMPT
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


_SPEAKER_MEMORY_TOOL_NAMES = (
    "speaker_memory_inspect",
    "speaker_memory_remember_name",
    "speaker_memory_confirm",
    "speaker_memory_reject",
    "speaker_memory_block_voice",
    "speaker_memory_unblock_voice",
    "speaker_memory_remember_fact",
    "speaker_memory_recall",
    "speaker_memory_forget",
)


def _function_tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _huggingvoice_system() -> str:
    return VOICE_SYSTEM_PROMPT


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


def test_gemma_speaker_memory_request_replaces_diluted_system_prompt():
    """Catch regressions that let the long voice prompt suppress native tool selection."""

    controller = FakeController()
    seen: list[dict] = []
    verbose_system = _huggingvoice_system()
    caller_system = "Never reveal credentials."
    with _client(controller, seen) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemma-4-e2b",
                "messages": [
                    {"role": "system", "content": caller_system},
                    {"role": "system", "content": verbose_system},
                    {
                        "role": "user",
                        "content": (
                            '<huggingvoice_speaker_context>{"speaker_ref":"sr_test",'
                            '"state":"unknown"}</huggingvoice_speaker_context>\n'
                            "Меня зовут Марат. Запомни моё имя."
                        ),
                    },
                ],
                "tools": [_function_tool(name) for name in _SPEAKER_MEMORY_TOOL_NAMES],
                "tool_choice": "auto",
            },
        )

    assert response.status_code == 200
    assert seen[0]["messages"][0] == {"role": "system", "content": caller_system}
    assert seen[0]["messages"][1] == {
        "role": "system",
        "content": (
            "Ты голосовой ассистент. Работа с памятью голосов HuggingVoice: "
            "после явного представления обязательно вызови "
            "speaker_memory_remember_name. Используй speaker_ref только из доверенного "
            "контекста. Не говори, что запомнил, пока инструмент не выполнен."
        ),
    }


def test_gemma_huggingvoice_web_superset_adds_only_available_tool_rules():
    controller = FakeController()
    seen: list[dict] = []
    tools = [
        *[_function_tool(name) for name in _SPEAKER_MEMORY_TOOL_NAMES],
        _function_tool("web_search"),
    ]

    with _client(controller, seen) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemma-4-e2b",
                "messages": [{"role": "system", "content": _huggingvoice_system()}],
                "tools": tools,
            },
        )

    assert response.status_code == 200
    compact = seen[0]["messages"][0]["content"]
    assert "speaker_memory_remember_name" in compact
    assert "web_search" in compact
    assert "camera_snapshot" not in compact


def test_gemma_huggingvoice_camera_superset_adds_all_available_tool_rules():
    controller = FakeController()
    seen: list[dict] = []
    tools = [
        *[_function_tool(name) for name in _SPEAKER_MEMORY_TOOL_NAMES],
        _function_tool("web_search"),
        _function_tool("camera_snapshot"),
    ]

    with _client(controller, seen) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemma-4-e2b",
                "messages": [{"role": "system", "content": _huggingvoice_system()}],
                "tools": tools,
            },
        )

    assert response.status_code == 200
    compact = seen[0]["messages"][0]["content"]
    assert "speaker_memory_remember_name" in compact
    assert "web_search" in compact
    assert "camera_snapshot" in compact


def test_speaker_memory_prompt_rewrite_is_limited_to_gemma_and_memory_only_tools():
    original_system = "Keep this caller policy unchanged."
    memory_tool = _function_tool("speaker_memory_remember_name")
    foreign_tool = _function_tool("get_weather")
    full_memory_tools = [_function_tool(name) for name in _SPEAKER_MEMORY_TOOL_NAMES]

    for model, tools in (
        ("ornith-35b", [memory_tool]),
        ("gemma-4-e2b", [memory_tool, foreign_tool]),
        ("gemma-4-e2b", [*full_memory_tools, foreign_tool]),
        (
            "gemma-4-e2b",
            [*full_memory_tools, _function_tool("web_search"), foreign_tool],
        ),
        ("gemma-4-e2b", full_memory_tools),
    ):
        controller = FakeController()
        seen: list[dict] = []
        with _client(controller, seen) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": original_system}],
                    "tools": tools,
                },
            )

        assert response.status_code == 200
        assert seen[0]["messages"] == [{"role": "system", "content": original_system}]


def test_marker_like_huggingvoice_prompt_with_extra_policy_is_not_rewritten():
    controller = FakeController()
    seen: list[dict] = []
    original = _huggingvoice_system() + "\nNever reveal tenant credentials."

    with _client(controller, seen) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemma-4-e2b",
                "messages": [{"role": "system", "content": original}],
                "tools": [
                    *[_function_tool(name) for name in _SPEAKER_MEMORY_TOOL_NAMES],
                    _function_tool("web_search"),
                ],
            },
        )

    assert response.status_code == 200
    assert seen[0]["messages"] == [{"role": "system", "content": original}]


def test_huggingvoice_prompt_rewrite_rejects_malformed_or_unknown_tool_sets():
    full_memory_tools = [_function_tool(name) for name in _SPEAKER_MEMORY_TOOL_NAMES]
    variants = (
        [*full_memory_tools, full_memory_tools[0]],
        [*full_memory_tools, _function_tool("speaker_memory_unknown")],
        [*full_memory_tools, {"type": "function"}],
        [*full_memory_tools[:-1], _function_tool("web_search")],
        [*full_memory_tools, _function_tool("web_search"), _function_tool("get_weather")],
    )

    for tools in variants:
        controller = FakeController()
        seen: list[dict] = []
        with _client(controller, seen) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemma-4-e2b",
                    "messages": [
                        {"role": "system", "content": _huggingvoice_system()}
                    ],
                    "tools": tools,
                },
            )

        assert response.status_code == 200
        assert seen[0]["messages"] == [
            {"role": "system", "content": _huggingvoice_system()}
        ]


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
