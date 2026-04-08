from __future__ import annotations

from app.tasks.base_task import BaseTask


MediumTask = BaseTask(name="anomaly_medium", dataset_file="transactions_medium.json")
