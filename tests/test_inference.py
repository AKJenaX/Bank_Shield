import os
import unittest
from io import StringIO
from unittest.mock import patch


class FakeEnv:
    def __init__(self) -> None:
        self._step_called = 0

    def reset(self, task_name: str):
        return {
            "observation": {
                "prompt": f"Task={task_name}. Decide for the next transaction.",
                "transaction": {"amount": 1200, "currency": "USD", "merchant": "X", "country": "US"},
            }
        }

    def step(self, action):
        self._step_called += 1
        # One-step episode; reward normalized to [0,1]
        return {
            "observation": {"prompt": "done", "transaction": {}},
            "reward": 0.8,
            "done": True,
            "error": None,
        }

    def state(self):
        return {"ok": True}


class FakeLLM:
    class _Chat:
        class _Completions:
            def create(self, **kwargs):
                class _R:
                    text = '{"decision":"allow","rationale":"Looks consistent with profile."}'

                return _R()

        completions = _Completions()

    chat = _Chat()


class TestInferenceEpisode(unittest.TestCase):
    def test_runs_one_episode_and_logs(self):
        # Import inside test so path/env patches apply cleanly in Cursor.
        from inference import run_episode

        fake_env = FakeEnv()
        fake_llm = FakeLLM()

        buf = StringIO()
        with patch("sys.stdout", buf):
            result = run_episode(
                task_name="easy",
                env=fake_env,  # type: ignore[arg-type]
                llm=fake_llm,  # type: ignore[arg-type]
                model_name="test-model",
                env_name="http://test-env",
                max_steps=10,
            )

        out = buf.getvalue()
        self.assertIn("[START]", out)
        self.assertIn("[STEP]", out)
        self.assertIn("[END]", out)
        self.assertGreaterEqual(result.steps, 1)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()

