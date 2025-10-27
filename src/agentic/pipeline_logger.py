import datetime
import json
from pathlib import Path
from typing import Any


class PipelineLogger:
    """Lightweight logger for main pipeline results and performance metrics.

    Provides structured logging for evaluation pipeline runs, including system tests,
    evaluation results, comparison metrics, and performance data. Automatically
    creates timestamped output directories and saves results in JSON format.

    Attributes:
        pipeline_name (str): Name identifier for the pipeline.
        run_id (str): Unique identifier for this specific run.
        results (dict): Structured dictionary containing all logged data including
            pipeline_info, system_test, evaluation_results, comparison_results,
            and performance_metrics.
        output_dir (Path): Directory path for saving output files.
    """

    def __init__(self, pipeline_name: str = "kaggle_evaluation", run_id: str = "") -> None:
        """Initialize the PipelineLogger with specified identifiers.

        Args:
            pipeline_name (str, optional): Name of the pipeline being logged.
                Defaults to "kaggle_evaluation".
            run_id (str, optional): Unique identifier for this run.
                Defaults to "".
        """
        self.pipeline_name = pipeline_name
        self.run_id = run_id
        self.results = {
            "pipeline_info": {
                "name": pipeline_name,
                "start_time": datetime.datetime.now().isoformat(),
            },
            "system_test": {},
            "evaluation_results": {},
            "comparison_results": {},
            "performance_metrics": {},
        }

        # Create output directory
        self.output_dir = Path(f"pipeline_results/{self.run_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_system_test(self, test_results: dict[str, Any]) -> None:
        """Log multi-agent system test results.

        Args:
            test_results (dict[str, Any]): Dictionary containing system test metrics
                and validation results.
        """
        self.results["system_test"] = test_results

    def log_evaluation_results(self, approach: str, chunk_count: int, doc_count: int, total_count: int) -> None:
        """Log evaluation results for a specific approach.

        Args:
            approach (str): Name of the evaluation approach (e.g., "multi_agent").
            chunk_count (int): Number of chunk ranking entries produced.
            doc_count (int): Number of document ranking entries produced.
            total_count (int): Total number of submission entries.
        """
        if "approaches" not in self.results["evaluation_results"]:
            self.results["evaluation_results"]["approaches"] = {}

        self.results["evaluation_results"]["approaches"][approach] = {
            "chunk_ranking_entries": chunk_count,
            "document_ranking_entries": doc_count,
            "total_entries": total_count,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def log_comparison(self, multi_agent_total: int, original_total: int, difference: int) -> None:
        """Log comparison results between different approaches.

        Args:
            multi_agent_total (int): Total entries from multi-agent approach.
            original_total (int): Total entries from original approach.
            difference (int): Difference between the two approaches.
        """
        self.results["comparison_results"] = {
            "multi_agent_entries": multi_agent_total,
            "original_entries": original_total,
            "difference": difference,
            "improvement_percentage": ((difference / original_total) * 100) if original_total > 0 else 0,
        }

    def log_performance_metrics(self, metrics: dict[str, Any]) -> None:
        """Log performance metrics for the evaluation run.

        Args:
            metrics (dict[str, Any]): Dictionary containing performance data
                such as duration, concurrency levels, and resource usage.
        """
        self.results["performance_metrics"] = metrics

    def save_results(self) -> str:
        """Save pipeline results to JSON file.

        Creates a timestamped JSON file containing all logged data from the
        evaluation run, including system tests, results, and performance metrics.

        Returns:
            str: Path to the saved JSON file.
        """
        self.results["pipeline_info"]["end_time"] = datetime.datetime.now().isoformat()

        # Save JSON format
        json_file = self.output_dir / f"{self.pipeline_name}_{self.run_id}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        return json_file
