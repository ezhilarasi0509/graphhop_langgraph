"""The LangChain-based client against a fake OpenAI-compatible server: no API key, no network."""
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from app import config, llm


class Handler(BaseHTTPRequestHandler):
    requests, fail_429, reject_json = [], 0, False

    def log_message(self, *a):
        pass

    def _send(self, code, body, headers=()):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests.append(body)
        if Handler.fail_429 > 0:
            Handler.fail_429 -= 1
            return self._send(429, {"error": {"message": "Rate limit reached. Please try again in 1s",
                                              "type": "tokens", "code": "rate_limit_exceeded"}}, [("retry-after", "1")])
        if Handler.reject_json and "response_format" in body:
            return self._send(400, {"error": {"message": "response_format unsupported", "type": "invalid_request_error"}})
        text = '{"a": 1}' if "response_format" in body else "ok"
        self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                         "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})


@pytest.fixture()
def server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(config, "LLM_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "test")
    Handler.requests, Handler.fail_429, Handler.reject_json = [], 0, False
    llm._model.cache_clear()
    yield Handler
    srv.shutdown()


def test_plain_call_counts_tokens(server):
    before = dict(llm.USAGE)
    assert llm.chat([{"role": "user", "content": "hi"}], model="openai/gpt-oss-120b") == "ok"
    assert llm.USAGE["tokens"] - before["tokens"] == 7 and llm.USAGE["calls"] - before["calls"] == 1
    body = server.requests[-1]
    assert body["messages"] == [{"role": "user", "content": "hi"}] and body["temperature"] == 0
    assert body["reasoning_effort"] == "low"          # extra_body reaches the provider for gpt-oss models


def test_json_mode_sends_response_format(server):
    assert llm.chat_json([{"role": "user", "content": "json please"}], model="m") == {"a": 1}
    assert server.requests[-1]["response_format"] == {"type": "json_object"}


def test_rate_limit_is_retried_and_wait_is_recorded(server, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    server.fail_429 = 2
    before = llm.WAITED[0]
    assert llm.chat([{"role": "user", "content": "hi"}], model="m") == "ok"
    assert len(server.requests) == 3 and llm.WAITED[0] - before >= 2


def test_json_mode_rejected_falls_back(server):
    server.reject_json = True
    assert llm.chat([{"role": "user", "content": "x"}], model="m", json_mode=True) == "ok"
    assert "response_format" not in server.requests[-1]


def test_cache_key_is_stable_for_dict_and_langchain_messages(server, tmp_path, monkeypatch):
    from langchain_core.messages import HumanMessage
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    llm.chat([{"role": "user", "content": "same"}], model="m", use_cache=True)
    llm.chat([HumanMessage(content="same")], model="m", use_cache=True)
    assert len(server.requests) == 1                    # second call is a cache hit
