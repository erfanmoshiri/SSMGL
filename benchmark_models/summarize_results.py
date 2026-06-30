"""
Aggregate benchmark results into summary tables.

Usage:
    python summarize_results.py [--results_dir results]
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path


def load_results(results_dir='results'):
    """Load all result JSON files."""
    results = []

    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return results

    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return results


def create_summary_table(results):
    """Create summary DataFrame from results."""
    if not results:
        return None

    # Extract key metrics
    rows = []
    for r in results:
        row = {
            'Model': r['model'],
            'Dataset': r['dataset'],
            'Recall@10': r.get('recall@10', 0),
            'Recall@20': r.get('recall@20', 0),
            'Recall@50': r.get('recall@50', 0),
            'NDCG@10': r.get('ndcg@10', 0),
            'NDCG@20': r.get('ndcg@20', 0),
            'NDCG@50': r.get('ndcg@50', 0),
            'Time(s)': r.get('train_time', 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by dataset and model
    df = df.sort_values(['Dataset', 'Model'])

    return df


def print_dataset_table(df, dataset):
    """Print formatted table for a single dataset."""
    dataset_df = df[df['Dataset'] == dataset].copy()

    if dataset_df.empty:
        return

    print(f"\n{'='*80}")
    print(f"{dataset.upper()} Dataset".center(80))
    print('='*80)

    # Drop dataset column for display
    display_df = dataset_df.drop('Dataset', axis=1)

    # Format floats
    for col in ['Recall@10', 'Recall@20', 'Recall@50', 'NDCG@10', 'NDCG@20', 'NDCG@50']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')

    display_df['Time(s)'] = display_df['Time(s)'].apply(lambda x: f'{x:.2f}')

    print(display_df.to_string(index=False))


def save_latex_table(df, output_file='results_table.tex'):
    """Save results as LaTeX table."""
    # Pivot for LaTeX format
    latex_tables = []

    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset].copy()

        # Select columns for LaTeX
        latex_df = dataset_df[['Model', 'Recall@10', 'Recall@20', 'Recall@50',
                                 'NDCG@10', 'NDCG@20', 'NDCG@50']].copy()

        # Format as percentages
        for col in latex_df.columns:
            if col != 'Model':
                latex_df[col] = latex_df[col].apply(lambda x: f'{x*100:.2f}')

        latex_str = latex_df.to_latex(index=False, escape=False,
                                        caption=f'Results on {dataset.upper()}',
                                        label=f'tab:{dataset}')
        latex_tables.append(latex_str)

    with open(output_file, 'w') as f:
        f.write('\n\n'.join(latex_tables))

    print(f"\nLaTeX tables saved to: {output_file}")


def save_csv(df, output_file='results_summary.csv'):
    """Save results as CSV."""
    df.to_csv(output_file, index=False)
    print(f"CSV saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Summarize benchmark results')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing result JSON files')
    parser.add_argument('--save_csv', action='store_true', default=True,
                        help='Save summary as CSV')
    parser.add_argument('--save_latex', action='store_true', default=False,
                        help='Save summary as LaTeX table')

    args = parser.parse_args()

    print("="*80)
    print("Benchmark Results Summary".center(80))
    print("="*80)

    # Load results
    results = load_results(args.results_dir)

    if not results:
        print("\nNo results found!")
        return

    print(f"\nLoaded {len(results)} result files")

    # Create summary table
    df = create_summary_table(results)

    if df is None or df.empty:
        print("No valid results to summarize!")
        return

    # Print tables per dataset
    for dataset in sorted(df['Dataset'].unique()):
        print_dataset_table(df, dataset)

    # Overall best models
    print(f"\n{'='*80}")
    print("Overall Best Models (by Recall@50)".center(80))
    print('='*80)

    for dataset in sorted(df['Dataset'].unique()):
        dataset_df = df[df['Dataset'] == dataset]
        best_model = dataset_df.loc[dataset_df['Recall@50'].idxmax()]
        print(f"{dataset.upper():10s}: {best_model['Model']:15s} "
              f"(Recall@50: {best_model['Recall@50']:.4f}, "
              f"NDCG@50: {best_model['NDCG@50']:.4f})")

    # Save outputs
    print(f"\n{'='*80}")

    if args.save_csv:
        save_csv(df, os.path.join(args.results_dir, 'summary.csv'))

    if args.save_latex:
        save_latex_table(df, os.path.join(args.results_dir, 'summary.tex'))

    print("="*80)


if __name__ == '__main__':
    main()
