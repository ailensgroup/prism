import json
from pathlib import Path

import pandas as pd


def extract_metrics(metrics_path: str) -> dict:
    """Extract relevant metrics from metrics.json file."""
    try:
        with open(metrics_path) as f:
            data = json.load(f)

        summary = data.get("summary", {})

        return {
            "input_tokens": summary.get("total_input_tokens", 0),
            "output_tokens": summary.get("total_output_tokens", 0),
            "processing_time": summary.get("total_processing_time_seconds", 0),
            "estimated_cost": summary.get("estimated_total_cost_usd", 0),
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading {metrics_path}: {e}")
        return None


def parse_folder_name(folder_name: str) -> tuple[str, str, int]:
    """Parse folder name to extract model, eval mode, and ICL information."""
    parts = folder_name.split("_")
    openai_model = parts[0] if parts else ""
    eval_mode_map = {
        "inContext": "In Context",
        "singleStore": "Single Store",
        "sharedStore": "Shared Store",
        "oracle": "Oracle",
    }
    eval_mode_raw = parts[1] if len(parts) > 1 else ""
    eval_mode = eval_mode_map.get(eval_mode_raw, eval_mode_raw)

    icl = 0
    for part in parts:
        if "icl" in part.lower():
            if "true" in part.lower():
                icl = 9
            elif "false" in part.lower():
                icl = 0
            break

    return openai_model, eval_mode, icl


def generate_excel_from_directory(base_directory: str, output_file: str = "results.xlsx") -> None:
    """Read through directory structure and generate Excel file.

    Args:
        base_directory: Path to the base directory (e.g., 'v1')
        output_file: Name of the output Excel file
    """
    base_path = Path(base_directory)

    if not base_path.exists():
        print(f"Directory {base_directory} does not exist!")
        return

    prompt_version = base_path.name
    data_rows = []
    for folder in base_path.iterdir():
        if folder.is_dir():
            openai_model, eval_mode, icl = parse_folder_name(folder.name)
            metrics_path = folder / "metrics.json"

            if metrics_path.exists():
                metrics = extract_metrics(metrics_path)

                if metrics:
                    row = {
                        "Prompt Version": prompt_version,
                        "OpenAI Model": openai_model,
                        "Eval Mode": eval_mode,
                        "ICL": icl,
                        "Input Token": metrics["input_tokens"],
                        "Output Token": metrics["output_tokens"],
                        "Processing Time": metrics["processing_time"],
                        "Estimated Cost (USD)": metrics["estimated_cost"],
                    }
                    data_rows.append(row)
                    print(f"Processed: {folder.name}")
            else:
                print(f"Warning: metrics.json not found in {folder.name}")

    if data_rows:
        df = pd.DataFrame(data_rows)
        model_order = {"gpt-4.1": 1, "gpt-5-mini": 2, "gpt-5": 3}
        eval_mode_order = {"Single Store": 1, "Shared Store": 2, "In Context": 3, "Oracle": 4}

        df["model_sort"] = df["OpenAI Model"].map(model_order).fillna(999)
        df["eval_mode_sort"] = df["Eval Mode"].map(eval_mode_order).fillna(999)

        df = df.sort_values(["ICL", "model_sort", "eval_mode_sort"])
        df = df.drop(columns=["model_sort", "eval_mode_sort"])

        df.to_excel(output_file, index=False, sheet_name="Results")
        print(f"\nExcel file '{output_file}' created successfully!")
        print(f"Total rows: {len(df)}")

        print("\nPreview of data:")
        print(df.head())
    else:
        print("No data found to export!")


if __name__ == "__main__":
    directory_path = "token_analysis/baseline_False/v4"
    output_filename = "evaluation_results.xlsx"
    generate_excel_from_directory(directory_path, output_filename)
