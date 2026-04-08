from __future__ import annotations

from app.tasks.base_task import BaseTask


HardTask = BaseTask(name="anomaly_hard", dataset_file="transactions_hard.json")
