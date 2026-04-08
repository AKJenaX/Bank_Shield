import unittest

from app.env.transaction_env import TransactionEnvironment


class TestEnvironmentInterface(unittest.TestCase):
    def test_env_openenv_interface_and_three_tasks(self):
        env = TransactionEnvironment(data_dir="data")
        for task in ("anomaly_easy", "anomaly_medium", "anomaly_hard"):
            result = env.reset(task)
            self.assertEqual(result.observation.task_id, task)
            self.assertGreaterEqual(result.reward.value, 0.0)
            self.assertLessEqual(result.reward.value, 1.0)

    def test_env_step_reward_and_done_progression(self):
        env = TransactionEnvironment(data_dir="data")
        result = env.reset("anomaly_easy")
        self.assertFalse(result.done)

        seen_done = False
        for _ in range(50):
            step_result = env.step('{"decision":"allow","rationale":"test"}')
            self.assertGreaterEqual(step_result.reward.value, 0.0)
            self.assertLessEqual(step_result.reward.value, 1.0)
            if step_result.done:
                seen_done = True
                break
        self.assertTrue(seen_done)


if __name__ == "__main__":
    unittest.main()
