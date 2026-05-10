"""Benchmark task definitions for agent SDK comparisons."""

from .basic_tasks import (
    BenchmarkTask,
    all_tasks,
    code_generation_task,
    data_parsing_task,
    file_read_write_task,
    get_task,
    multi_step_reasoning_task,
    task_ids,
    web_search_task,
)

__all__ = [
    "BenchmarkTask",
    "all_tasks",
    "code_generation_task",
    "data_parsing_task",
    "file_read_write_task",
    "get_task",
    "multi_step_reasoning_task",
    "task_ids",
    "web_search_task",
]
