import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aiFeatures.MistralClient import MistralClient


def load_benchmark_data(filepath: str = "benchmark.json") -> List[Dict[str, Any]]:
    """Load benchmark test cases from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_benchmark_item(client: MistralClient, item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Process a single benchmark item through the AI pipeline
    Returns enriched item with AI analysis results
    """
    print(f"\n{'='*80}")
    print(f"Processing item {index + 1}: {item['text'][:80]}...")
    print(f"{'='*80}")
    
    result = {
        "original": item.copy(),
        "ai_analysis": {},
        "evaluation": {}
    }
    
    try:
        # Step 1: List event facts (extract claims)
        print("\n[1/2] Extracting claims with list_event_facts()...")
        events_json, list_impacts = client.list_event_facts(item['text'])
        
        try:
            events = json.loads(events_json)
            result["ai_analysis"]["extracted_events"] = events
            result["ai_analysis"]["list_impacts"] = list_impacts
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse events JSON: {e}")
            result["ai_analysis"]["extracted_events"] = []
            result["ai_analysis"]["list_impacts"] = list_impacts
            result["ai_analysis"]["list_error"] = str(e)
        
        # Step 2: Analyze the main claim
        print("\n[2/2] Analyzing claim with analyze_event()...")
        analysis_json, analyze_impacts = client.analyze_event(item['text'])
        
        try:
            # Clean up the JSON response if it has markdown formatting
            cleaned_json = analysis_json.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            cleaned_json = cleaned_json.strip()
            
            analysis = json.loads(cleaned_json)
            result["ai_analysis"]["analysis"] = analysis
            result["ai_analysis"]["analyze_impacts"] = analyze_impacts
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse analysis JSON: {e}")
            print(f"Raw response: {analysis_json[:500]}")
            result["ai_analysis"]["analysis"] = {"raw_response": analysis_json}
            result["ai_analysis"]["analyze_impacts"] = analyze_impacts
            result["ai_analysis"]["analysis_error"] = str(e)
        
        # Step 3: Evaluate accuracy
        result["evaluation"] = evaluate_result(item, result["ai_analysis"])
        
    except Exception as e:
        print(f"Error processing item: {e}")
        result["error"] = str(e)
        result["evaluation"] = {
            "correct": False,
            "ai_score": 0,
            "has_references": False,
            "valid_sources": False,
            "error": str(e)
        }
    
    return result


def evaluate_result(original: Dict[str, Any], ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the AI's performance on this item
    Returns evaluation metrics
    """
    evaluation = {
        "correct": False,
        "ai_score": 0,
        "has_references": False,
        "valid_sources": False,
        "reference_count": 0,
        "wikipedia_refs": 0,
        "rag_refs": 0,
        "expected_label": original.get("label"),
        "expected_score": original.get("score"),
        "expected_sources": original.get("sources", [])
    }
    
    # Extract AI analysis details
    analysis = ai_analysis.get("analysis", {})
    if isinstance(analysis, dict) and "score" in analysis:
        evaluation["ai_score"] = analysis.get("score", 0)
        references = analysis.get("references", [])
        
        # Check if references exist
        evaluation["has_references"] = len(references) > 0
        evaluation["reference_count"] = len(references)
        
        # Count reference types
        for ref in references:
            ref_lower = ref.lower()
            if "wikipedia" in ref_lower:
                evaluation["wikipedia_refs"] += 1
            if ".pdf" in ref_lower or "document:" in ref_lower:
                evaluation["rag_refs"] += 1
        
        # Validate sources (must be wikipedia or RAG)
        evaluation["valid_sources"] = evaluation["wikipedia_refs"] > 0 or evaluation["rag_refs"] > 0
        
        # Determine correctness based on multiple criteria
        # True positive: label is true and AI score > 1
        # True negative: label is false and AI score <= 1
        expected_label = original.get("label")
        if expected_label == "true":
            evaluation["correct"] = evaluation["ai_score"] >= 2
        elif expected_label == "false":
            evaluation["correct"] = evaluation["ai_score"] <= 1
        
        # Store additional metrics
        evaluation["accuracy_text"] = analysis.get("accuracy", "")
        evaluation["biases_text"] = analysis.get("biases", "")
        evaluation["contextualization_text"] = analysis.get("contextualization", "")
    
    return evaluation


def analyze_benchmark_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze aggregate statistics from benchmark results
    """
    total = len(results)
    correct = sum(1 for r in results if r["evaluation"].get("correct", False))
    
    # Accuracy metrics
    with_refs = [r for r in results if r["evaluation"].get("has_references", False)]
    without_refs = [r for r in results if not r["evaluation"].get("has_references", False)]
    valid_sources = [r for r in results if r["evaluation"].get("valid_sources", False)]
    
    # Score distribution
    scores = [r["evaluation"].get("ai_score", 0) for r in results]
    
    # Reference statistics
    total_refs = sum(r["evaluation"].get("reference_count", 0) for r in results)
    total_wikipedia = sum(r["evaluation"].get("wikipedia_refs", 0) for r in results)
    total_rag = sum(r["evaluation"].get("rag_refs", 0) for r in results)
    
    # Environmental impact
    total_gwp = sum(
        r["ai_analysis"].get("analyze_impacts", {}).get("gwp", {}).get("max", 0) + 
        r["ai_analysis"].get("list_impacts", {}).get("gwp", {}).get("max", 0)
        for r in results
    )
    total_tokens = sum(
        r["ai_analysis"].get("analyze_impacts", {}).get("total_input_tokens", 0) + 
        r["ai_analysis"].get("analyze_impacts", {}).get("total_output_tokens", 0) +
        r["ai_analysis"].get("list_impacts", {}).get("total_input_tokens", 0) + 
        r["ai_analysis"].get("list_impacts", {}).get("total_output_tokens", 0)
        for r in results
    )
    
    return {
        "total_items": total,
        "correct_predictions": correct,
        "accuracy": correct / total if total > 0 else 0,
        "items_with_references": len(with_refs),
        "items_without_references": len(without_refs),
        "items_with_valid_sources": len(valid_sources),
        "accuracy_with_refs": sum(1 for r in with_refs if r["evaluation"].get("correct", False)) / len(with_refs) if with_refs else 0,
        "accuracy_without_refs": sum(1 for r in without_refs if r["evaluation"].get("correct", False)) / len(without_refs) if without_refs else 0,
        "accuracy_with_valid_sources": sum(1 for r in valid_sources if r["evaluation"].get("correct", False)) / len(valid_sources) if valid_sources else 0,
        "score_distribution": {
            "0": sum(1 for s in scores if s == 0),
            "1": sum(1 for s in scores if s == 1),
            "2": sum(1 for s in scores if s == 2),
            "3": sum(1 for s in scores if s == 3)
        },
        "total_references": total_refs,
        "wikipedia_references": total_wikipedia,
        "rag_references": total_rag,
        "total_gwp_impact": total_gwp,
        "total_tokens": total_tokens
    }


def plot_benchmark_results(results: List[Dict[str, Any]], statistics: Dict[str, Any], output_dir: str = "benchmark_results"):
    """
    Create visualizations of benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Overall Accuracy
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Benchmark Analysis Results', fontsize=16, fontweight='bold')
    
    # 1.1: Overall Accuracy
    ax = axes[0, 0]
    categories = ['Overall', 'With\nReferences', 'Without\nReferences', 'Valid\nSources']
    accuracies = [
        statistics['accuracy'] * 100,
        statistics['accuracy_with_refs'] * 100,
        statistics['accuracy_without_refs'] * 100,
        statistics['accuracy_with_valid_sources'] * 100
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    bars = ax.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy by Category', fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 1.2: Score Distribution
    ax = axes[0, 1]
    scores = list(statistics['score_distribution'].values())
    score_labels = ['Score 0', 'Score 1', 'Score 2', 'Score 3']
    colors_scores = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
    bars = ax.bar(score_labels, scores, color=colors_scores, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Items', fontweight='bold')
    ax.set_title('AI Score Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # 1.3: Reference Sources
    ax = axes[1, 0]
    ref_categories = ['Items with\nReferences', 'Items without\nReferences', 'Valid Sources']
    ref_counts = [
        statistics['items_with_references'],
        statistics['items_without_references'],
        statistics['items_with_valid_sources']
    ]
    colors_refs = ['#27ae60', '#c0392b', '#8e44ad']
    bars = ax.bar(ref_categories, ref_counts, color=colors_refs, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Items', fontweight='bold')
    ax.set_title('Reference Coverage', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # 1.4: Reference Type Distribution
    ax = axes[1, 1]
    ref_types = ['Wikipedia', 'RAG', 'Total']
    ref_type_counts = [
        statistics['wikipedia_references'],
        statistics['rag_references'],
        statistics['total_references']
    ]
    colors_types = ['#3498db', '#e67e22', '#95a5a6']
    bars = ax.bar(ref_types, ref_type_counts, color=colors_types, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of References', fontweight='bold')
    ax.set_title('Reference Types', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_overview.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {os.path.join(output_dir, 'benchmark_overview.png')}")
    
    # Figure 2: Accuracy by AI Score
    fig, ax = plt.subplots(figsize=(10, 6))
    
    score_accuracies = []
    score_counts = []
    for score in range(4):
        items_with_score = [r for r in results if r["evaluation"].get("ai_score") == score]
        if items_with_score:
            correct = sum(1 for r in items_with_score if r["evaluation"].get("correct", False))
            score_accuracies.append((correct / len(items_with_score)) * 100)
            score_counts.append(len(items_with_score))
        else:
            score_accuracies.append(0)
            score_counts.append(0)
    
    x = np.arange(4)
    bars = ax.bar(x, score_accuracies, color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71'], 
                   alpha=0.7, edgecolor='black')
    ax.set_xlabel('AI Score', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Prediction Accuracy by AI Score', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Score {i}\n(n={score_counts[i]})' for i in range(4)])
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_score.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'accuracy_by_score.png')}")
    
    plt.close('all')


def save_results(results: List[Dict[str, Any]], statistics: Dict[str, Any], output_file: str = "benchmark_results.json"):
    """
    Save complete benchmark results to JSON file
    """
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_items": statistics["total_items"],
            "overall_accuracy": statistics["accuracy"]
        },
        "statistics": statistics,
        "detailed_results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved detailed results to: {output_file}")


def print_summary(statistics: Dict[str, Any]):
    """
    Print a summary of benchmark results
    """
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"\nTotal Items:              {statistics['total_items']}")
    print(f"Correct Predictions:      {statistics['correct_predictions']}")
    print(f"Overall Accuracy:         {statistics['accuracy']*100:.2f}%")
    print(f"\nWith References:          {statistics['items_with_references']} items ({statistics['accuracy_with_refs']*100:.2f}% accuracy)")
    print(f"Without References:       {statistics['items_without_references']} items ({statistics['accuracy_without_refs']*100:.2f}% accuracy)")
    print(f"With Valid Sources:       {statistics['items_with_valid_sources']} items ({statistics['accuracy_with_valid_sources']*100:.2f}% accuracy)")
    print(f"\nScore Distribution:")
    for score, count in statistics['score_distribution'].items():
        print(f"  Score {score}:              {count} items")
    print(f"\nReferences:")
    print(f"  Total:                  {statistics['total_references']}")
    print(f"  Wikipedia:              {statistics['wikipedia_references']}")
    print(f"  RAG:                    {statistics['rag_references']}")
    print(f"\nEnvironmental Impact:")
    print(f"  Total GWP:              {statistics['total_gwp_impact']:.6f} kgCO2eq")
    print(f"  Total Tokens:           {statistics['total_tokens']:,}")
    print("="*80 + "\n")


def main():
    """
    Main benchmark execution function
    """
    print("="*80)
    print("STARTING BENCHMARK ANALYSIS")
    print("="*80)
    
    # Initialize client
    print("\nInitializing Mistral Client...")
    client = MistralClient()
    
    # Load benchmark data
    print("Loading benchmark data from benchmark.json...")
    benchmark_data = load_benchmark_data()
    print(f"Loaded {len(benchmark_data)} test items")
    
    # Process each item
    results = []
    for i, item in enumerate(benchmark_data):
        result = process_benchmark_item(client, item, i)
        results.append(result)
        
        # Show progress
        if (i + 1) % 5 == 0 or (i + 1) == len(benchmark_data):
            print(f"\nProgress: {i + 1}/{len(benchmark_data)} items processed")
    
    # Analyze results
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    statistics = analyze_benchmark_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.json"
    save_results(results, statistics, results_file)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    results_dir = f"benchmark_results_{timestamp}"
    plot_benchmark_results(results, statistics, results_dir)
    
    # Print summary
    print_summary(statistics)
    
    print("✓ Benchmark analysis complete!")
    print(f"\nResults saved in:")
    print(f"  - {results_file}")
    print(f"  - {results_dir}/")


if __name__ == "__main__":
    main()