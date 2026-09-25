import os
import json
import csv
from typing import List, Dict, Any
import pandas as pd
from oct_bench.config import config
from oct_bench.data.schemas import MCQ
from .models import BaseModelEvaluator

class BenchmarkEvaluator:
    """
    Executes benchmark evaluations, tracks model predictions, calculates metrics,
    and exports summary reports in CSV format.
    """
    def __init__(self, model_evaluator: BaseModelEvaluator, model_name: str):
        self.evaluator = model_evaluator
        self.model_name = model_name
        self.reports_dir = config.paths.get("reports_dir", "./reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    def evaluate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Loads dataset, executes evaluation, saves detailed outputs and calculates accuracy splits.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"QA dataset not found at {dataset_path}")
            
        with open(dataset_path, "r") as f:
            raw_qa = json.load(f)
            
        print(f"Loaded {len(raw_qa)} questions for evaluation.")
        
        results = []
        correct_count = 0
        
        # Categorized metric collections
        dimensions_stats = {
            "Perception": {"correct": 0, "total": 0},
            "Cognition": {"correct": 0, "total": 0},
            "Reasoning": {"correct": 0, "total": 0}
        }
        
        tasks_stats = {}
        diseases_stats = {}
        
        for index, item in enumerate(raw_qa):
            qid = item["question_id"]
            task_id = item["task_id"]
            img_path = item["metadata"].get("rendered_image_path", "")
            
            # Map task ID to dimension
            dimension = "Reasoning"
            if task_id.startswith(("T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08")):
                dimension = "Perception"
            elif task_id.startswith(("T09", "T10", "T11", "T12", "T13", "T14", "T15", "T16")):
                dimension = "Cognition"
                
            disease = item["metadata"].get("disease_label", "NORMAL")
            
            # Run inference
            print(f"[{index+1}/{len(raw_qa)}] Running {self.model_name} on {qid} (Task {task_id})...")
            pred = self.evaluator.predict(img_path, item["question"], item["options"])
            
            is_correct = (pred == item["correct_answer"])
            if is_correct:
                correct_count += 1
                dimensions_stats[dimension]["correct"] += 1
                
            dimensions_stats[dimension]["total"] += 1
            
            # Sub-track by fine task ID
            if task_id not in tasks_stats:
                tasks_stats[task_id] = {"correct": 0, "total": 0}
            tasks_stats[task_id]["total"] += 1
            if is_correct:
                tasks_stats[task_id]["correct"] += 1
                
            # Sub-track by disease category
            if disease not in diseases_stats:
                diseases_stats[disease] = {"correct": 0, "total": 0}
            diseases_stats[disease]["total"] += 1
            if is_correct:
                diseases_stats[disease]["correct"] += 1

            results.append({
                "question_id": qid,
                "task_id": task_id,
                "dimension": dimension,
                "disease": disease,
                "question": item["question"],
                "options": str(item["options"]),
                "correct_answer": item["correct_answer"],
                "prediction": pred,
                "is_correct": is_correct
            })
            
        overall_acc = (correct_count / len(raw_qa)) * 100 if raw_qa else 0.0
        
        # Build evaluation report dictionary
        report = {
            "model_name": self.model_name,
            "overall_accuracy": overall_acc,
            "dimension_accuracies": {
                dim: (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
                for dim, stats in dimensions_stats.items()
            },
            "task_accuracies": {
                t: (stats["correct"] / stats["total"]) * 100
                for t, stats in tasks_stats.items()
            },
            "disease_accuracies": {
                d: (stats["correct"] / stats["total"]) * 100
                for d, stats in diseases_stats.items()
            }
        }
        
        # Save logs and summary report
        self._export_results(results, report)
        return report

    def _export_results(self, results: List[Dict], report: Dict):
        """
        Saves prediction details and metrics summaries to CSV / text reports.
        """
        model_slug = self.model_name.lower().replace("-", "_")
        
        # Save raw predictions CSV
        preds_csv_path = os.path.join(self.reports_dir, f"{model_slug}_predictions.csv")
        df_preds = pd.DataFrame(results)
        df_preds.to_csv(preds_csv_path, index=False)
        print(f"Saved raw predictions log at {preds_csv_path}")
        
        # Save summary report text
        summary_path = os.path.join(self.reports_dir, f"{model_slug}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"=== EVALUATION SUMMARY FOR MODEL: {report['model_name']} ===\n")
            f.write(f"Overall Accuracy: {report['overall_accuracy']:.2f}%\n\n")
            
            f.write("--- DIMENSION ACCURACIES ---\n")
            for dim, acc in report["dimension_accuracies"].items():
                f.write(f"{dim}: {acc:.2f}%\n")
                
            f.write("\n--- DISEASE ACCURACIES ---\n")
            for dis, acc in report["disease_accuracies"].items():
                f.write(f"{dis}: {acc:.2f}%\n")
                
            f.write("\n--- TASK-WISE ACCURACIES ---\n")
            for task, acc in sorted(report["task_accuracies"].items()):
                f.write(f"Task {task}: {acc:.2f}%\n")
                
        print(f"Saved evaluation text summary at {summary_path}")
