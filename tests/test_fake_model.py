"""The offline fake endpoint drives the real ChatOpenAI client through one investigation (control flow, not quality).

Run in the agent venv: agent\\venv\\Scripts\\python.exe -m unittest tests.test_fake_model -v
"""

import asyncio
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from langchain_openai import ChatOpenAI

from agent.config import AgentConfig
from agent.run import build_llm, investigate
from experiments import aggregate, batch
from experiments.fake_model import serve_in_background
from experiments.openrouter_audit import audit_run
from tests.agent_fakes import FakeTools


class FakeEndpointTests(unittest.TestCase):
    def test_scripted_investigation_through_the_openai_client(self):
        server, url = serve_in_background()
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as folder:
            config = AgentConfig.from_env({"MODEL_BASE_URL": url, "MODEL_ID": "offline-fake",
                                           "RUN_TAG": "offline_unit", "RESULTS_DIR": folder})
            tools = FakeTools()

            async def connect(config):
                llm = ChatOpenAI(model=config.model_id, api_key="offline", base_url=config.base_url,
                                 timeout=10, max_retries=0)
                return llm, tools.build()

            result = asyncio.run(investigate(config, "Assess current health", connect))
            trace = [json.loads(line) for line in
                     (Path(folder) / "offline_unit.trace.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_call_sequence"], ["query_logs", "get_metrics"])
        self.assertEqual(result["calls_used"], 4)
        self.assertEqual((result["confidence"], result["root_cause_id"]), ("inconclusive", "none"))
        self.assertGreater(result["input_tokens_total"], 0)  # usage parsed from the HTTP reply
        # Cost and generation identity survive the real client (OpenRouter-style body).
        self.assertEqual(result["cost_reported_calls"], 4)
        self.assertGreater(result["cost_usd_total"], 0)
        self.assertEqual(len(set(result["generation_ids"])), 4)
        self.assertTrue(all(gid.startswith("gen-offline-") for gid in result["generation_ids"]))
        self.assertEqual(result["served_models"], ["offline-fake"])
        self.assertEqual([name for name, _ in tools.invocations],
                         ["get_container_state", "get_metrics", "query_logs", "get_metrics"])
        self.assertTrue(all(e.get("attempts") == 1 for e in trace if e["event"] == "action_selected"))

    def test_pinned_routing_reaches_the_endpoint_and_passes_the_audit(self):
        server, url = serve_in_background()
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as folder:
            config = AgentConfig.from_env({
                "MODEL_BASE_URL": url, "MODEL_ID": "offline-fake", "RUN_TAG": "pinned_unit", "RESULTS_DIR": folder,
                "OPENROUTER_API_KEY": "offline-key", "MODEL_PROVIDER_ORDER": "offline-pin",
                "MODEL_ALLOW_FALLBACKS": "false", "MODEL_SEED": "7"})

            async def connect(config):
                return build_llm(config), FakeTools().build()

            result = asyncio.run(investigate(config, "Assess current health", connect))
            summary, errors = audit_run(Path(folder), "pinned_unit", url, config.api_key,
                                        config.model_settings.provider_order, sleep=lambda s: None)
            written = (Path(folder) / "pinned_unit.result.json").read_text(encoding="utf-8")
        self.assertEqual(errors, [])
        self.assertEqual(summary["providers_served"], ["offline-pin"])  # the fake echoes provider.order[0]
        self.assertEqual(summary["retrieved"], len(result["generation_ids"]))
        self.assertAlmostEqual(summary["total_cost_usd"], result["cost_usd_total"])
        self.assertEqual(json.loads(written)["model_settings"]["provider_order"], ["offline-pin"])
        self.assertNotIn("offline-key", written)


class ErrorReplyTests(unittest.TestCase):
    """OpenRouter-style refusals through the real client (build_llm, default max_retries)."""

    def serve(self, status: int, body: dict) -> tuple[str, list]:
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                requests.append(self.path)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}/v1", requests

    def run_agent(self, url):
        with tempfile.TemporaryDirectory() as folder:
            config = AgentConfig.from_env({"MODEL_BASE_URL": url, "MODEL_ID": "vendor/model", "RUN_TAG": "refused",
                                           "RESULTS_DIR": folder, "OPENROUTER_API_KEY": "offline-key"})

            async def connect(config):
                return build_llm(config), FakeTools().build()

            result = asyncio.run(investigate(config, "Assess current health", connect))
            written = {path.name for path in Path(folder).iterdir()}
        return config, result, written

    def test_a_200_with_an_error_body_is_a_classified_provider_error_with_artifacts(self):
        url, requests = self.serve(200, {"error": {"message": "Key limit exceeded", "code": 403}})
        config, result, written = self.run_agent(url)
        self.assertEqual(config.model_settings.max_retries, 1)
        self.assertEqual((result["status"], result["error_type"]), ("crashed", "ProviderErrorResponse"))
        self.assertEqual(written, {"refused.result.json", "refused.trace.jsonl"})
        self.assertEqual(aggregate.invalid_cause({"validity_errors": []}, result), "provider")
        self.assertTrue(batch.KEY_FAILURE.search(result["traceback_tail"]))  # its message stops the batch
        self.assertEqual(len(requests), 1)

    def test_a_key_limit_403_matches_the_batch_stop_and_is_not_retried(self):
        url, requests = self.serve(403, {"error": {"message": "Key limit exceeded (total limit)", "code": 403}})
        _, result, _ = self.run_agent(url)
        self.assertEqual(result["status"], "crashed")
        self.assertIn("Error code: 403", result["traceback_tail"])
        self.assertTrue(batch.KEY_FAILURE.search(result["traceback_tail"]))
        self.assertEqual(aggregate.invalid_cause({"validity_errors": []}, result), "provider")
        self.assertEqual(len(requests), 1)

    def test_a_429_is_retried_once_and_is_not_a_key_failure(self):
        url, requests = self.serve(429, {"error": {"message": "Rate limit exceeded: free-models-per-day", "code": 429}})
        _, result, _ = self.run_agent(url)
        self.assertIn("Error code: 429", result["traceback_tail"])
        self.assertIsNone(batch.KEY_FAILURE.search(result["traceback_tail"]))
        self.assertEqual(len(requests), 2)  # max_retries 1: the call and one retry


if __name__ == "__main__":
    unittest.main()
