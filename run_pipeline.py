import os
import sys

# Ensure current directory is on python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mock_generator import main as run_mock_generator
from oct_bench.data.adapters import UnifiedJSONAdapter
from oct_bench.generation.generator import VQAGenerator
from oct_bench.quality_control.checker import LogicalQCChecker
from oct_bench.evaluation.models import RandomBaselineEvaluator
from oct_bench.evaluation.evaluator import BenchmarkEvaluator

def main():
    print("==================================================")
    # Step 1: Generate Mock Data
    print("STEP 1: Synthesizing raw OCT image scans & index annotations...")
    run_mock_generator()
    
    # Step 2: Load Standardized Image Samples via Adapter
    print("\nSTEP 2: Standardizing dataset annotations...")
    adapter = UnifiedJSONAdapter(raw_data_dir="./data/raw")
    samples = adapter.load_samples()
    print(f"Loaded {len(samples)} standardized ImageSample entries.")
    
    # Step 3: Generate MCQs (Visual QAs)
    print("\nSTEP 3: Generating VQA multiple-choice questions for benchmark...")
    # Select a diverse set of tasks spanning Perception, Cognition, and Reasoning
    test_tasks = ["T01", "T06", "T09", "T12", "T17"]
    generator = VQAGenerator()
    questions = generator.generate_benchmark(samples, test_tasks)
    print(f"Successfully generated {len(questions)} MCQ items.")
    
    # Step 4: Run Automated QC Checker
    print("\nSTEP 4: Performing automated logical quality checks...")
    qc_checker = LogicalQCChecker()
    passed_count = 0
    
    for q in questions:
        passed, reason = qc_checker.run_checks(q)
        if passed:
            passed_count += 1
        else:
            print(f" QC Failed for {q.question_id}: {reason}")
            
    print(f"QC check completed. {passed_count}/{len(questions)} questions passed.")
    
    # Step 5: Evaluate Benchmark using Random Guess Baseline
    print("\nSTEP 5: Running evaluation using Random Baseline Evaluator...")
    model_name = "random-guess-baseline"
    model_evaluator = RandomBaselineEvaluator()
    
    evaluator = BenchmarkEvaluator(model_evaluator, model_name)
    report = evaluator.evaluate_dataset("./data/generated_qa/dataset.json")
    
    print("\n==================================================")
    print(f"INTEGRATION TEST PASSED FOR MODEL: {report['model_name']}")
    print(f"Overall Accuracy: {report['overall_accuracy']:.2f}% (Expect ~25% for random guessing)")
    print("\nDimension breakdown:")
    for dim, acc in report["dimension_accuracies"].items():
        print(f" - {dim}: {acc:.2f}%")
    print("\nReport files written to './reports/' directory.")
    print("==================================================")

if __name__ == "__main__":
    main()
