from main import MTEvaluator
import os
import matplotlib
import traceback
matplotlib.use('Agg')

def main():
    """Run a complete evaluation for thesis submission"""
    print("Starting comprehensive thesis evaluation...")
    
    evaluator = MTEvaluator(
        data_dir="data",
        results_dir="results/thesis_evaluation"
    )
    
    os.makedirs("results/evaluation", exist_ok=True)
    os.makedirs("results/evaluation/figures", exist_ok=True)
    
    try:
        print("\n==== RUNNING MAIN EVALUATION ====")
        results = evaluator.run_evaluation(
            translation_systems=["marian_mt", "deepl", "google_translate"], 
            evaluation_metrics=["bleu", "meteor", "chrf", "rouge"],
            target_language="es",
            num_samples=150,
            reference_system="deepl"
        )
        
        print("\n==== GENERATING ENHANCED REPORTS ====")
        evaluator.generate_report(results)
        
        print("\nGenerating back-translation evaluation...")
        evaluator.add_back_translation_evaluation()
        
        print("\nAnalyzing sentiment preservation...")
        evaluator.analyze_sentiment_preservation()
        
        print("\nAnalyzing entity preservation...")
        evaluator.analyze_entity_preservation()
        
        print("\nAnalyzing readability...")
        evaluator.analyze_readability()
        
        print("\nEvaluating cultural nuance preservation...")
        evaluator.evaluate_cultural_nuance_preservation()
        
        print("\nNormalizing metrics...")
        evaluator.normalize_metrics_across_domains()
        
        print("\nComputing unified score...")
        unified_score = evaluator.compute_unified_score()
        print(f"\nUnified scores: {unified_score.to_string()}")
        
        print("\n==== RUNNING DOMAIN-SPECIFIC EVALUATIONS ====")
        domain_results = evaluator.run_domain_specific_evaluation()
        
        for domain, score in domain_results.items():
            print(f"Domain {domain} unified score: {score}")
        
        print("\n==== GENERATING COMPREHENSIVE REPORT ====")
        report_path = evaluator.generate_comprehensive_report()
        print(f"Comprehensive report generated at: {report_path}")
        
        
        if validation["issues"]:
            print("\nPlease address the following issues before thesis submission:")
            for issue in validation["issues"]:
                print(f"  - {issue}")
        
        print("\nEvaluation complete! Check the results directory for all reports and figures.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
