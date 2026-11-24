"""Contracts that must hold before any model experiment starts."""

import asyncio
import unittest

from agent.config import AgentConfig
from agent.tools import MAX_OBSERVATION_CHARS, observe, usable_evidence


class AgentConfigTests(unittest.TestCase):
    def test_valid_run(self):
        config = AgentConfig.from_env({"MODEL_BASE_URL": "http://localhost:8000/v1",
                                       "MODEL_ID": "test-model", "RUN_TAG": "trial_001"})
        self.assertEqual(config.run_tag, "trial_001")

    def test_rejects_path_traversal(self):
        base = {"MODEL_BASE_URL": "http://localhost:8000/v1", "MODEL_ID": "test"}
        with self.assertRaisesRegex(ValueError, "RUN_TAG"):
            AgentConfig.from_env({**base, "RUN_TAG": "../outside"})

    def test_api_key_falls_back_to_openrouter_key_and_is_never_shown(self):
        base = {"MODEL_BASE_URL": "https://openrouter.ai/api/v1", "MODEL_ID": "openai/gpt-6-luna"}
        self.assertEqual(AgentConfig.from_env({**base, "OPENROUTER_API_KEY": "or-key"}).api_key, "or-key")
        both = AgentConfig.from_env({**base, "OPENROUTER_API_KEY": "or-key", "MODEL_API_KEY": "model-key"})
        self.assertEqual(both.api_key, "model-key")
        self.assertEqual(AgentConfig.from_env(base).api_key, "local")
        self.assertNotIn("model-key", repr(both))

    def test_routing_settings_are_validated(self):
        base = {"MODEL_BASE_URL": "https://openrouter.ai/api/v1", "MODEL_ID": "z-ai/glm-5.3"}
        settings = AgentConfig.from_env({**base, "MODEL_PROVIDER_ORDER": "wafer, ", "MODEL_QUANTIZATIONS": "fp4",
                                         "MODEL_ALLOW_FALLBACKS": "FALSE", "MODEL_SEED": "20261123"}).model_settings
        self.assertEqual(settings.provider_routing(), {"allow_fallbacks": False, "order": ["wafer"],
                                                       "quantizations": ["fp4"]})
        self.assertEqual(settings.seed, 20261123)
        for name, value in (("MODEL_ALLOW_FALLBACKS", "maybe"), ("MODEL_SEED", "-1"),
                            ("MODEL_DATA_COLLECTION", "sometimes")):
            with self.assertRaisesRegex(ValueError, name):
                AgentConfig.from_env({**base, name: value})

    def test_rejects_non_model_endpoint(self):
        with self.assertRaisesRegex(ValueError, "MODEL_BASE_URL"):
            AgentConfig.from_env({"MODEL_BASE_URL": "http://localhost:8000/api/ps",
                                  "MODEL_ID": "test"})


class ObservationTests(unittest.TestCase):
    def test_caps_long_output_and_records_time(self):
        class Tool:
            async def ainvoke(self, args):
                return "x" * (MAX_OBSERVATION_CHARS + 500)

        observation = asyncio.run(observe(Tool(), {}))
        self.assertIn("TRUNCATED", observation["result"])
        self.assertIn("+00:00", observation["observed_at"])
        self.assertLess(len(observation["result"]), MAX_OBSERVATION_CHARS + 100)

    def test_failed_observation_cannot_support_a_diagnosis(self):
        class BrokenTool:
            async def ainvoke(self, args):
                raise RuntimeError("unavailable")

        observation = asyncio.run(observe(BrokenTool(), {}))
        self.assertFalse(usable_evidence(observation))
        self.assertFalse(usable_evidence({"result": "(0 of 10 lines matched filter)"}))


if __name__ == "__main__":
    unittest.main()
