import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

AZURE_PRICING = {
    "gpt-4.1": {"input": 2 / 1000000, "output": 8 / 1000000},  # per 1M token
    "gpt-5-mini": {"input": 0.25 / 1000000, "output": 2 / 1000000},
    "gpt-5": {"input": 1.25 / 1000000, "output": 10 / 1000000},
    "gpt-5.1": {"input": 1.25 / 1000000, "output": 10 / 1000000},
    "gpt-5.2": {"input": 1.75 / 1000000, "output": 14 / 1000000},
    "deepseek-v3.2": {"input": 0.58 / 1000000, "output": 1.68 / 1000000},
    "gpt-oss-120b": {"input": 0.15 / 1000000, "output": 0.60 / 1000000},
    "grok-4-20-reasoning": {"input": 3 / 1000000, "output": 15 / 1000000},
    "llama-4-maverick-17b-128e-instruct-fp8": {"input": 0.25 / 1000000, "output": 1 / 1000000},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on token usage."""
    for model_key, pricing in AZURE_PRICING.items():
        if model_key in model.lower():
            cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
            return round(cost, 6)

    return (input_tokens * AZURE_PRICING["gpt-5"]["input"]) + (output_tokens * AZURE_PRICING["gpt-5"]["output"])


@dataclass
class APICallMetrics:
    """Metrics for a single API call."""

    start_time: str
    end_time: str
    timestamp: str
    question_id: str
    question: str
    eval_mode: str
    model: str
    prompt_version: str
    use_icl: bool

    input_prompt: str
    input_tokens: int
    output_response: str
    output_raw_response: str
    output_tokens: int
    total_tokens: int

    processing_time_seconds: float

    answer: str
    retrieved_documents_count: int

    api_call_type: str
    success: bool
    error_message: str | None = None

    estimated_cost_usd: float | None = None


class MetricsTracker:
    """Track and save metrics for API calls."""

    def __init__(self, output_dir: str = "./metrics") -> None:
        """Initialize metrics tracker.

        Args:
            output_dir: Directory to save metrics JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_metrics: list[APICallMetrics] = []

        print(f"Metrics tracker initialized. Output: {self.output_dir}")

    def create_run_directory(
        self,
        model: str,
        eval_mode: str = "new",
        prompt_version: str = "v_unknown",
        use_icl: bool = True,
        run_idx: str = "0",
    ) -> Path:
        """Create a directory for this specific run."""
        run_name = f"{model}_{eval_mode}_v{prompt_version}_icl{use_icl}_run{run_idx}"
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    def record_metric(self, metric: APICallMetrics) -> None:
        """Record a single API call metric."""
        self.current_run_metrics.append(metric)

    def save_run_metrics(self, run_dir: Path, filename: str = "metrics.json") -> Path:
        """Save API call metric."""
        output_file = run_dir / filename
        metrics_data = [asdict(m) for m in self.current_run_metrics]
        summary = self._calculate_summary(self.current_run_metrics)
        output = {
            "run_info": {
                "total_calls": len(self.current_run_metrics),
                "timestamp": datetime.now().isoformat(),
                "output_directory": str(run_dir),
            },
            "summary": summary,
            "detailed_metrics": metrics_data,
        }

        # Write to a temp file first, then atomically replace the real file
        with tempfile.NamedTemporaryFile(mode="w", dir=run_dir, suffix=".tmp", delete=False) as tmp:
            json.dump(output, tmp, indent=2)
            tmp_path = Path(tmp.name)

        tmp_path.replace(output_file)  # atomic on most OS/filesystems

        print(f"Saved {len(metrics_data)} metrics to {output_file}")
        return output_file

    def _calculate_summary(self, metrics: list[APICallMetrics]) -> dict:
        """Calculate summary statistics."""
        if not metrics:
            return {}

        total_input_tokens = sum(m.input_tokens for m in metrics)
        total_output_tokens = sum(m.output_tokens for m in metrics)
        total_tokens = sum(m.total_tokens for m in metrics)

        total_time = sum(m.processing_time_seconds for m in metrics)
        avg_time = total_time / len(metrics)

        successful_calls = sum(1 for m in metrics if m.success)
        failed_calls = len(metrics) - successful_calls

        total_cost = sum(m.estimated_cost_usd for m in metrics if m.estimated_cost_usd)

        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "average_input_tokens": total_input_tokens / len(metrics),
            "average_output_tokens": total_output_tokens / len(metrics),
            "total_processing_time_seconds": total_time,
            "average_processing_time_seconds": avg_time,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "estimated_total_cost_usd": total_cost,
        }

    def clear_current_run(self) -> None:
        """Clear metrics for starting a new run."""
        self.current_run_metrics = []

    def export_summary_csv(self, run_dir: Path, filename: str = "summary.csv") -> None:
        """Export summary statistics to CSV."""
        if not self.current_run_metrics:
            return

        metrics_dicts = [asdict(m) for m in self.current_run_metrics]
        df = pd.DataFrame(metrics_dicts)

        output_file = run_dir / filename
        df.to_csv(output_file, index=False)

        print(f"📊 Saved summary CSV to {output_file}")
