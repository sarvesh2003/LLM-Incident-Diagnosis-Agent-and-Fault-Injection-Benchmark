"""Deterministic OpenAI-compatible endpoint for offline orchestration checks. THIS IS NOT A MODEL.

It follows a fixed script and mimics OpenRouter's generation ids, usage.cost and /generation lookup; never score its runs.
Usage: python -m experiments.fake_model --port 18999, then MODEL_BASE_URL=http://127.0.0.1:18999/v1 MODEL_ID=offline-fake
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

MODEL_ID = "offline-fake"
FINAL_REPORT = {
    "root_cause_id": "none",
    "summary": "Offline control-flow fixture; no diagnosis was attempted.",
    "confidence": "inconclusive",
    "reasoning": "Scripted reply from experiments/fake_model.py, not a model.",
    "cited_evidence_indices": [0, 1],
}
FAKE_PRICE_PER_TOKEN = {"input": 0.10 / 1e6, "output": 0.50 / 1e6}
PROVIDER_NAME = "offline-fake"
GENERATIONS: dict[str, dict] = {}  # id -> /generation record, for the audit endpoint
# Calls remaining -> scripted tool call; otherwise conclude with an "inconclusive" report.
SCRIPTED_CALLS = {
    13: ("query_logs", {"service": "go-enricher", "since": "5m"}),
    12: ("get_metrics", {"promql": "rate(fraud_enricher_messages_processed_total[1m])", "window": "5m"}),
}


def text_of(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def reply(request: dict) -> dict:
    """The scripted chat-completion response for one request body."""
    prompt = " ".join(text_of(message) for message in request.get("messages", []))
    message: dict = {"role": "assistant", "content": ""}
    if "Respond with ONLY" in prompt:
        message["content"] = json.dumps(FINAL_REPORT)
        finish = "stop"
    else:
        remaining = re.search(r"Tool calls remaining: (\d+) of", prompt)
        name, args = SCRIPTED_CALLS.get(int(remaining.group(1)) if remaining else -1,
                                        ("ConcludeInvestigation", {"reason": "offline script complete"}))
        message["tool_calls"] = [{"id": f"call_{name}", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}]
        finish = "tool_calls"
    prompt_tokens, completion_tokens = max(1, len(prompt) // 4), 20
    cost = round(prompt_tokens * FAKE_PRICE_PER_TOKEN["input"] + completion_tokens * FAKE_PRICE_PER_TOKEN["output"], 10)
    generation_id = "gen-offline-" + uuid.uuid4().hex[:16]
    model = request.get("model", MODEL_ID)
    order = (request.get("provider") or {}).get("order") or [PROVIDER_NAME]
    GENERATIONS[generation_id] = {
        "id": generation_id, "model": model, "provider_name": order[0], "total_cost": cost,
        "tokens_prompt": prompt_tokens, "tokens_completion": completion_tokens,
        "native_tokens_reasoning": 0, "finish_reason": finish,
        "provider_responses": [{"provider_name": order[0], "status": 200}],
    }
    return {"id": generation_id, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens, "cost": cost,
                      "completion_tokens_details": {"reasoning_tokens": 0}}}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        elif path.rstrip("/") == "/v1/generation":
            record = GENERATIONS.get(parse_qs(query).get("id", [""])[0])
            if record:
                self._send(200, {"data": record})
            else:
                self._send(404, {"error": {"message": "generation not found"}})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        self._send(200, reply(body))

    def log_message(self, format, *args):  # keep test output quiet
        pass


def serve_in_background(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start on 127.0.0.1 (port 0 = any free port); returns the server and its /v1 URL."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=18999)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"offline fake model (not a model) at http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
