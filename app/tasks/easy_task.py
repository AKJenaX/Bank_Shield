from __future__ import annotations

from app.tasks.base_task import BaseTask


EasyTask = BaseTask(name="anomaly_easy", dataset_file="transactions_easy.json")
