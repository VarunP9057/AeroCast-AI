import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

REPORTS = [
    {
        "dataset": "Passenger",
        "experiment": "Original Model Comparison",
        "file": "reports/evaluation/model_comparison.csv",
        "evaluation_type": "Test Set Comparison"
    },
    {
        "dataset": "Passenger",
        "experiment": "Improved ANN Experiment",
        "file": "reports/ann_experiment/ann_experiment_metrics.csv",
        "evaluation_type": "Experiment Comparison"
    },
    {
        "dataset": "Aircraft",
        "experiment": "Original Model Comparison",
        "file": "reports/aircraft_forecasting/aircraft_model_comparison.csv",
        "evaluation_type": "Test Set Comparison"
    },
    {
        "dataset": "Aircraft",
        "experiment": "Experiment 2",
        "file": "reports/aircraft_forecasting_experiment2/aircraft_experiment2_model_comparison.csv",
        "evaluation_type": "Experiment Comparison"
    },
    {
        "dataset": "Aircraft",
        "experiment": "Experiment 3",
        "file": "reports/aircraft_forecasting_experiment3/aircraft_experiment3_metrics.csv",
        "evaluation_type": "ANN Experiment"
    },
    {
        "dataset": "Cargo",
        "experiment": "Original Model Comparison",
        "file": "reports/cargo_forecasting/cargo_model_comparison.csv",
        "evaluation_type": "Test Set Comparison"
    },
    {
        "dataset": "Cargo",
        "experiment": "Advanced Model Experiment",
        "file": "reports/cargo_forecasting_experiment2/cargo_experiment2_model_comparison.csv",
        "evaluation_type": "Experiment Comparison"
    },
    {
        "dataset": "Cargo",
        "experiment": "Experiment 3 ANN Validation",
        "file": "reports/cargo_forecasting_experiment3/cargo_experiment3_validation_results.csv",
        "evaluation_type": "Walk-Forward Validation"
    },
    {
        "dataset": "Cargo",
        "experiment": "Experiment 4 Algorithm Validation",
        "file": "reports/cargo_forecasting_experiment4/cargo_experiment4_validation_results.csv",
        "evaluation_type": "Walk-Forward Validation"
    },
    {
        "dataset": "Operational Pressure",
        "experiment": "Original Model Comparison",
        "file": "reports/operational_pressure/operational_pressure_model_comparison.csv",
        "evaluation_type": "Test Set Comparison"
    },
    {
        "dataset": "Operational Pressure",
        "experiment": "ANN Validation",
        "file": "reports/operational_pressure/operational_pressure_ann_validation_summary.csv",
        "evaluation_type": "Walk-Forward Validation"
    },
    {
        "dataset": "Terminal",
        "experiment": "Original Model Comparison",
        "file": "reports/terminal_forecasting/terminal_model_comparison.csv",
        "evaluation_type": "Test Set Comparison"
    },
    {
        "dataset": "Terminal",
        "experiment": "Experiment 3 Validation",
        "file": "reports/terminal_forecasting_experiment3/terminal_experiment3_summary.csv",
        "evaluation_type": "Walk-Forward Validation"
    }
]


def get_value(row, possible_names):
    for name in possible_names:
        value = row.get(name)

        if value is not None and str(value).strip() != "":
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

    return None


def read_report(report):
    file_path = BASE_DIR / report["file"]

    if not file_path.exists():
        return []

    results = []

    with open(file_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            model_name = row.get("Model", "Unknown Model")
            terminal = row.get("Terminal")

            if terminal:
                model_name = f"{terminal} - {model_name}"

            result = {
                "dataset": report["dataset"],
                "experiment": report["experiment"],
                "evaluation_type": report["evaluation_type"],
                "model": model_name,
                "mae": get_value(row, ["MAE", "Average_MAE"]),
                "rmse": get_value(row, ["RMSE", "Average_RMSE"]),
                "r2": get_value(row, ["R2", "R2 Score", "Average_R2"]),
                "mape": get_value(row, ["MAPE", "MAPE (%)", "Average_MAPE"]),
                "wape": get_value(row, ["WAPE", "WAPE (%)", "Average_WAPE"]),
                "rank": get_value(row, ["Rank"])
            }

            results.append(result)

    return results


def get_model_lab_results(dataset=None):
    results = []
    missing_reports = []

    for report in REPORTS:
        if dataset and report["dataset"].lower() != dataset.lower():
            continue

        report_results = read_report(report)

        if not report_results:
            missing_reports.append(report["file"])

        results.extend(report_results)

    return {
        "total_results": len(results),
        "datasets": sorted(set(item["dataset"] for item in results)),
        "missing_reports": missing_reports,
        "results": results
    }


def get_model_lab_overview():
    data = get_model_lab_results()

    dataset_summary = {}

    for dataset in data["datasets"]:
        dataset_results = [
            item for item in data["results"]
            if item["dataset"] == dataset
        ]

        dataset_summary[dataset] = {
            "experiment_count": len(set(
                item["experiment"] for item in dataset_results
            )),
            "model_result_count": len(dataset_results),
            "ann_result_count": sum(
                "ann" in item["model"].lower()
                for item in dataset_results
            )
        }

    model_files = list((BASE_DIR / "models").rglob("*.keras"))

    return {
        "total_results": data["total_results"],
        "total_datasets": len(data["datasets"]),
        "trained_ann_artifacts": len(model_files),
        "datasets": dataset_summary,
        "missing_reports": data["missing_reports"]
    }
