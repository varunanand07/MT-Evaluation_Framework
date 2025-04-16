from src.main import MTEvaluator
import os
import numpy as np

os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

evaluator = MTEvaluator(data_dir="data", results_dir="results")

print("Loading translation model...")
evaluator.load_translation_model("marian_mt", "en", "es")

ENHANCED_MODE = True

if ENHANCED_MODE:
    print("\nRunning domain-specific evaluation on larger dataset...")
    domain_results = evaluator.run_domain_specific_evaluation(
        domains=['medical', 'technical', 'legal', 'general'],
        systems=["marian_mt", "deepl", "google_translate"],
        metrics=["bleu", "meteor", "chrf", "rouge"]
    )
    
    print("\nEnhanced evaluation complete!")
    
else:
    print("Running main evaluation...")
    results = evaluator.run_evaluation(
        translation_systems=["marian_mt", "deepl", "google_translate"], 
        evaluation_metrics=["bleu", "meteor", "chrf", "rouge"],
        target_language="es",
        num_samples=150,
        reference_system="deepl"
    )

    print("Generating reports...")
    report_path = evaluator.generate_report(results)
    side_by_side_path = evaluator.generate_side_by_side_html()
    print(f"Evaluation report generated at: {report_path}")
    print(f"Side-by-side comparison generated at: {side_by_side_path}")

    print("\nRunning back-translation analysis...")
    back_translations = evaluator.add_back_translation_evaluation()

    print("\nAnalyzing sentiment preservation...")
    sentiment_results = evaluator.analyze_sentiment_preservation()

    print("\nAnalyzing named entity preservation...")
    entity_results = evaluator.analyze_entity_preservation()

    print("\nAnalyzing readability...")
    readability_results = evaluator.analyze_readability()

    print("\nEvaluating cultural nuance preservation...")
    cultural_results = evaluator.evaluate_cultural_nuance_preservation()

    print("\nPerforming statistical validation...")
    statistical_validation = evaluator.perform_statistical_validation()

    print("\nComputing unified meaning preservation score...")
    unified_score = evaluator.compute_unified_score()

    print("\nGenerating comprehensive report...")
    comprehensive_report = evaluator.generate_comprehensive_report()

    print("\nUpdating comprehensive report with actual findings...")
    evaluator.update_comprehensive_report_findings()

    print("\nValidating output files...")
    if evaluator.validate_outputs():
        print("✓ All expected output files validated successfully")
    else:
        print("⚠ Some output files are missing or corrupted. Review warnings above.")

    print("\nRunning domain-specific evaluations...")
    domain_results = evaluator.run_domain_specific_evaluation()

print("\nAll analyses complete! Your evaluation framework is now thesis-ready.")
