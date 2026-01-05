"""
Medical QA - Visualization Module
=============================================

This module provides functions for visualizing data and analysis results.
It enforces a consistent Blue/Azure color scheme and handles plot saving.
"""

import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from cycler import cycler
import seaborn as sns

# =============================================================================
# Plot Saving
# =============================================================================

def _save_plot(save_path: Optional[str], default_filename: str) -> None:
    """
    Internal helper to handle plot saving logic.
    Supports both directory paths (appends default_filename) and file paths.
    """
    if not save_path:
        return
        
    # Check if save_path looks like a file (has extension)
    if save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
        # It's a file path
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        save_path_full = save_path
    else:
        # It's a directory
        os.makedirs(save_path, exist_ok=True)
        save_path_full = os.path.join(save_path, default_filename)
        
    plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path_full}")

# =============================================================================
# Configuration & Style
# =============================================================================

def set_plot_style():
    """
    Sets the matplotlib and seaborn style for the project.
    
    Returns:
        tuple: (selected_palette, div_cmap, gradient_cmap)
    """
    plt.style.use('ggplot')
    
    # Custom Blue/Azure color cycle
    selected_colors = [
        "#003f5c",  # deep blue
        "#2f4b7c",  # dark blue
        "#665191",  # purple-blue
        "#a05195",  # purple
        "#d45087",  # pink
        "#f95d6a",  # coral
        "#ff7c43",  # orange
        "#ffa600",  # yellow
    ]
    
    # Alternative purely blue palette for consistency
    blue_palette = [
        "#0D47A1", # 900
        "#1565C0", # 800
        "#1976D2", # 700
        "#1E88E5", # 600
        "#2196F3", # 500
        "#42A5F5", # 400
        "#64B5F6", # 300
        "#90CAF9", # 200
    ]
    
    selected_color_cycle = cycler('color', blue_palette)
    selected_palette = sns.color_palette(blue_palette)
    
    mpl.rcParams['axes.prop_cycle'] = selected_color_cycle
    sns.set_palette(selected_palette)
    
    # Custom colormaps
    div_cmap = LinearSegmentedColormap.from_list(
        "nc_div", ["#194BFF", "#FFFFFF", "#0685A8"], N=256
    )
    gradient_cmap = LinearSegmentedColormap.from_list(
        "nc_gradient", ["#E3F2FD", "#0D47A1"], N=256
    )
    
    return selected_palette, div_cmap, gradient_cmap

# =============================================================================
# Exploratory Data Analysis (Visualization)
# =============================================================================

def plot_distribution(data, title, xlabel, ylabel, bins=50, save_path='output', filename='distribution.png'):
    """
    Plots a simple distribution histogram with KDE.
    
    Args:
        data (pd.Series or list): Input data.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        bins (int): Number of bins.
        save_path (str): Directory to save figure.
        filename (str): Filename for saved figure.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if save_path:
        _save_plot(save_path, filename)
        
    plt.show()

def plot_rag_retrieval_stats(results_data, save_path='output'):
    """
    Plots retrieval distances per query using a violin plot with labeled top sources.
    
    Args:
        results_data (list of dict): List containing {'Query': str, 'Distance': float, 'Rank': int}
        save_path (str): Directory to save figure.
    """
    plt.figure(figsize=(12, 8))
    
    df = pd.DataFrame(results_data)
    
    # Violin plot for density
    sns.violinplot(x='Query', y='Distance', data=df, inner=None, color='lightblue', linewidth=1.5)
    
    # Swarm plot for individual points (avoids overlap)
    sns.swarmplot(x='Query', y='Distance', data=df, color='#004c6d', size=8, alpha=0.8)
    
    # Label top 3 ranks
    queries = df['Query'].unique()
    query_to_x = {q: i for i, q in enumerate(queries)}
    
    for _, row in df.iterrows():
        if row['Rank'] <= 3:
            x = query_to_x[row['Query']]
            y = row['Distance']
            # Offset text slightly to the right
            plt.text(x + 0.1, y, str(row['Rank']), fontsize=10, color='black', fontweight='bold', va='center')

    plt.title('Retrieval Confidence per Query')
    plt.ylabel('L2 Distance (Lower is Better)')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    
    # Threshold line in BLUE
    plt.axhline(y=0.80, color='blue', linestyle='--', alpha=0.7, label='Abstain Threshold (0.80)')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        _save_plot(save_path, 'retrieval_distances_per_query.png')
        
    plt.show()

def plot_evaluation_metrics(plot_df, scenario_order, save_path=None):
    """
    Plots violin plots with connected points for Faithfulness and Relevance metrics.
    
    Args:
        plot_df (pd.DataFrame): DataFrame containing evaluation results (filtered for non-abstentions).
        scenario_order (list): List of scenario names in the desired order.
        save_path (str, optional): Path to save the plot.
    """
    # Set up the figure for vertical stacking (2 rows, 1 column)
    plt.figure(figsize=(12, 14))

    def _plot_metric_with_connections(metric_name, ax_index):
        plt.subplot(2, 1, ax_index)
        
        # 1. Violin Plot (showing median with inner='quartile')
        sns.violinplot(
            data=plot_df, 
            x="Scenario", 
            y=metric_name, 
            order=scenario_order,
            hue="Scenario", 
            legend=False, 
            palette="Blues", 
            inner="quartile", # Show quartiles/median lines inside violin
            linewidth=1.5,
            alpha=0.3
        )
        
        # 2. Strip Plot (Points)
        sns.stripplot(
            data=plot_df, 
            x="Scenario", 
            y=metric_name, 
            order=scenario_order,
            color="#0D47A1", 
            alpha=0.6, 
            jitter=0.05, # Small jitter to see overlapping points
            size=6
        )
        
        # 3. Connect paired points with lines
        # Pivot data to get scores for each query across scenarios
        # Use groupby and unstack to handle potential duplicates safely
        pivot_data = plot_df.groupby(['Query', 'Scenario'])[metric_name].mean().unstack()
        
        # Map scenarios to x-coordinates (0, 1, 2)
        x_map = {name: i for i, name in enumerate(scenario_order)}
        
        for query in pivot_data.index:
            # Get scores for this query
            scores = pivot_data.loc[query]
            
            # Collect (x, y) pairs
            xs = []
            ys = []
            for scenario in scenario_order:
                if scenario in scores and not pd.isna(scores[scenario]):
                    xs.append(x_map[scenario])
                    ys.append(scores[scenario])
            
            # Plot line if we have at least 2 points to connect
            if len(xs) > 1:
                plt.plot(xs, ys, color='gray', alpha=0.2, linewidth=0.8, zorder=0)

        plt.title(f"{metric_name} Distribution (Higher is Better)", fontsize=14)
        plt.ylim(-0.5, 5.5) # Give some space for text
        plt.grid(axis='y', alpha=0.3)
        
        # Add Median annotation text at the top
        medians = plot_df.groupby("Scenario")[metric_name].median()
        for i, scenario in enumerate(scenario_order):
            if scenario in medians:
                plt.text(i, 5.2, f"Med: {medians[scenario]:.1f}", ha='center', fontweight='bold', color='#0D47A1')

    # Plot Faithfulness
    _plot_metric_with_connections("Faithfulness", 1)

    # Plot Relevance
    _plot_metric_with_connections("Relevance", 2)

    plt.tight_layout()
    
    if save_path:
        _save_plot(save_path, "evaluation_violin_plots_paired.png")
    
    plt.show()

def plot_faithfulness_gap_distribution(results_df, save_path=None):
    """
    Plots the distribution of Faithfulness differences (RAG - No RAG).
    Highlights the median gap.
    """
    # Determine available RAG scenario
    available_scenarios = results_df['Scenario'].unique()
    if "RAG (No Abstain)" in available_scenarios:
        rag_scenario = "RAG (No Abstain)"
    elif "RAG + Abstain" in available_scenarios:
        rag_scenario = "RAG + Abstain"
    else:
        print("Warning: No RAG scenario found for Gap Analysis.")
        return

    norag_scenario = "No RAG (LLM Only)"

    rag_df = results_df[results_df['Scenario'] == rag_scenario].set_index('Query')
    norag_df = results_df[results_df['Scenario'] == norag_scenario].set_index('Query')

    # Combine into a comparison dataframe
    comparison = pd.DataFrame({
        'RAG_Faithfulness': rag_df['Faithfulness'],
        'NoRAG_Faithfulness': norag_df['Faithfulness']
    }).dropna()

    # Calculate Gap
    comparison['Gap'] = comparison['RAG_Faithfulness'] - comparison['NoRAG_Faithfulness']
    
    mean_gap = comparison['Gap'].mean()

    plt.figure(figsize=(10, 6))
    
    # Histogram with KDE
    # Use a nice blue color matching the theme
    sns.histplot(comparison['Gap'], kde=True, bins=15, color="#1976D2", alpha=0.6, edgecolor='white')
    
    # Vertical line for Mean
    plt.axvline(mean_gap, color='#0D47A1', linestyle='--', linewidth=2, label=f'Mean Gap: {mean_gap:.2f}')
    
    plt.title("Distribution of Faithfulness Gap (RAG - No RAG)", fontsize=14)
    plt.xlabel("Faithfulness Difference (Positive = RAG is better)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend()
    # plt.grid(axis='y', alpha=0.3) # ggplot already has grid
    
    if save_path:
        _save_plot(save_path, "faithfulness_gap_histogram.png")
        
    plt.show()

def plot_hallucination_rates(results_df, save_path=None):
    """
    Plots the Hallucination Rate (Faithfulness <= 2.0) for each scenario as a bar chart.
    """
    # Define scenarios and order
    all_scenarios = ["RAG + Abstain", "RAG (No Abstain)", "No RAG (LLM Only)"]
    
    # Filter to only scenarios present in the data
    available_scenarios = set(results_df['Scenario'].unique())
    scenario_order = [s for s in all_scenarios if s in available_scenarios]
    
    rates = []
    for scenario in scenario_order:
        scenario_data = results_df[results_df['Scenario'] == scenario]
        n_total = len(scenario_data)
        if n_total == 0:
            rate = 0
        else:
            # Count hallucinations (Faithfulness <= 2.0)
            n_hallucinations = len(scenario_data[scenario_data['Faithfulness'] <= 2.0])
            rate = (n_hallucinations / n_total) * 100
            
        rates.append({'Scenario': scenario, 'Hallucination Rate (%)': rate})
    
    rates_df = pd.DataFrame(rates)
    
    plt.figure(figsize=(10, 6))
    
    # Use Blue palette to maintain brand consistency
    # Lighter blue for lower risk (RAG + Abstain), Darker blue for higher risk (No RAG)
    custom_palette = ["#90CAF9", "#42A5F5", "#0D47A1"]
    
    ax = sns.barplot(
        data=rates_df, 
        x='Scenario', 
        y='Hallucination Rate (%)', 
        order=scenario_order,
        palette=custom_palette
    )
    
    plt.title("Hallucination Rate by Scenario (Lower is Better)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Hallucination Rate (%)", fontsize=12, fontweight='bold')
    plt.xlabel("", fontsize=12)
    
    # Set y-limit with some headroom
    max_rate = rates_df['Hallucination Rate (%)'].max() if not rates_df.empty else 0
    plt.ylim(0, max(5, max_rate * 1.2)) 
    plt.grid(axis='y', alpha=0.3)
    
    # Add labels on top of bars
    for i, v in enumerate(rates_df['Hallucination Rate (%)']):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    if save_path:
        _save_plot(save_path, "hallucination_rates_bar.png")
        
    plt.show()

def plot_score_distribution_stacked(results_df, metric="Faithfulness", save_path=None):
    """
    Plots a 100% stacked bar chart showing the distribution of scores (1-5) for each scenario.
    This is often clearer than violin plots for discrete Likert-scale data.
    """
    # Define scenarios and order
    all_scenarios = ["RAG + Abstain", "RAG (No Abstain)", "No RAG (LLM Only)"]
    available_scenarios = [s for s in all_scenarios if s in results_df['Scenario'].unique()]
    
    # Prepare data: Count occurrences of each score (1-5) per scenario
    score_counts = results_df.groupby(['Scenario', metric]).size().unstack(fill_value=0)
    
    # Ensure all scores 1-5 are present
    for i in range(1, 6):
        if i not in score_counts.columns:
            score_counts[i] = 0
    score_counts = score_counts[[1, 2, 3, 4, 5]] # Reorder
    
    # Convert to percentages
    score_pcts = score_counts.div(score_counts.sum(axis=1), axis=0) * 100
    
    # Reorder rows based on scenario_order
    score_pcts = score_pcts.reindex(available_scenarios)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Custom Blue scale for 1-5 scores (Light -> Dark)
    # 1=Lightest Blue, 5=Darkest Blue
    colors = ["#90CAF9", "#42A5F5", "#2196F3", "#1976D2", "#0D47A1"]
    
    ax = score_pcts.plot(
        kind='barh', 
        stacked=True, 
        color=colors, 
        figsize=(10, 5),
        width=0.7,
        edgecolor='white'
    )
    
    plt.title(f"{metric} Score Distribution by Scenario", fontsize=14)
    plt.xlabel("Percentage of Answers (%)", fontsize=12)
    plt.ylabel("")
    
    # Add percentage labels on the bars
    for c in ax.containers:
        # Optional: Filter out small labels
        labels = [f'{v.get_width():.0f}%' if v.get_width() > 4 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=10)
    
    # Move legend outside
    plt.legend(title="Score", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        _save_plot(save_path, f"{metric.lower()}_distribution_stacked.png")
        
    plt.show()

def plot_rouge_scores(valid_results, save_path=None):
    """
    Plots the ROUGE-L scores with error bars.
    """
    # Calculate Statistics
    stats = valid_results.groupby('Scenario')['ROUGE-L'].agg(['mean', 'sem', 'count', 'std']).reset_index()
    print("\nROUGE-L Stats (Answer vs. Context):")
    print(stats)

    # Plotting
    plt.figure(figsize=(10, 6))
    # Removed local style override to maintain global ggplot style
    
    # Define colors matching the Hallucination Rate plot (Light Blue, Medium Blue, Dark Blue)
    custom_palette = {
        "RAG + Abstain": "#90CAF9",       # Light Blue (200)
        "RAG (No Abstain)": "#42A5F5",    # Medium Blue (400)
        "No RAG (LLM Only)": "#1565C0"    # Dark Blue (800)
    }
    
    # Bar plot with Error Bars (Standard Error)
    ax = sns.barplot(
        data=valid_results, 
        x='Scenario', 
        y='ROUGE-L', 
        hue='Scenario',
        errorbar='se', 
        palette=custom_palette,
        capsize=0.1,
        order=["RAG + Abstain", "RAG (No Abstain)", "No RAG (LLM Only)"],
        legend=False
    )

    plt.title('Baseline Metric: ROUGE-L Score (Answer vs. Context)\n(Higher is better, indicates content overlap)', fontsize=14)
    plt.ylabel('ROUGE-L Score', fontsize=12)
    plt.xlabel('Scenario', fontsize=12)
    
    # Improve Y-axis (Dynamic limit based on max value)
    max_y = 0
    if not stats.empty:
        # Calculate max height including error bar
        stats['upper'] = stats['mean'] + stats['sem']
        max_y = stats['upper'].max()
        plt.ylim(0, max_y * 1.2)
    
    # Add labels manually above error bars
    scenarios_order = ["RAG + Abstain", "RAG (No Abstain)", "No RAG (LLM Only)"]
    
    for i, scenario in enumerate(scenarios_order):
        scenario_stats = stats[stats['Scenario'] == scenario]
        if not scenario_stats.empty:
            mean_val = scenario_stats['mean'].values[0]
            sem_val = scenario_stats['sem'].values[0]
            if pd.isna(sem_val): sem_val = 0
            
            # Position text above the error bar
            text_y = mean_val + sem_val
            
            ax.text(i, text_y, f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold', color='black', fontsize=10)

    plt.tight_layout()
    
    if save_path:
        _save_plot(save_path, "rouge_scores.png")
        
    plt.show()

def plot_faithfulness_scores(plot_df, save_path=None):
    """
    Plots the Average Faithfulness Score with error bars.
    """
    print("\nGenerating Faithfulness Score Plot...")

    # Calculate stats for label placement
    stats = plot_df.groupby('Scenario')['Faithfulness'].agg(['mean', 'sem']).reset_index()

    plt.figure(figsize=(10, 6))
    # Removed local style override to maintain global ggplot style

    # Custom palette (Light Blue -> Dark Blue)
    custom_palette = {
        "RAG + Abstain": "#90CAF9",       # Light Blue (200)
        "RAG (No Abstain)": "#42A5F5",    # Medium Blue (400)
        "No RAG (LLM Only)": "#1565C0"    # Dark Blue (800)
    }

    # Bar plot with Error Bars (Standard Error)
    ax = sns.barplot(
        data=plot_df, 
        x='Scenario', 
        y='Faithfulness', 
        hue='Scenario',
        errorbar='se', 
        palette=custom_palette,
        capsize=0.1,
        order=["RAG + Abstain", "RAG (No Abstain)", "No RAG (LLM Only)"],
        legend=False
    )

    plt.title('Average Faithfulness Score (1-5)\n(Higher is Better - Measures Grounding in Context)', fontsize=14)
    plt.ylabel('Faithfulness Score', fontsize=12)
    plt.xlabel('Scenario', fontsize=12)

    # Dynamic Y-limit (Score is 1-5)
    plt.ylim(0, 5.5)

    # Add labels manually above error bars
    scenarios_order = ["RAG + Abstain", "RAG (No Abstain)", "No RAG (LLM Only)"]
    
    for i, scenario in enumerate(scenarios_order):
        scenario_stats = stats[stats['Scenario'] == scenario]
        if not scenario_stats.empty:
            mean_val = scenario_stats['mean'].values[0]
            sem_val = scenario_stats['sem'].values[0]
            if pd.isna(sem_val): sem_val = 0
            
            # Position text above the error bar
            text_y = mean_val + sem_val
            
            ax.text(i, text_y, f'{mean_val:.2f}', 
                    ha='center', va='bottom', fontweight='bold', color='black', fontsize=10)

    plt.tight_layout()
    
    if save_path:
        _save_plot(save_path, 'faithfulness_mean_score.png')
        
    plt.show()

def plot_comparative_gains(results_df, save_path=None):
    """
    Plots comparative gains of RAG scenarios vs No RAG baseline.
    Metrics:
    1. Hallucinations Fixed (Count): Number of No-RAG hallucinations (<=2.0) that were fixed (>2.0).
    2. Queries Improved (Count): Number of queries where RAG score > No RAG score.
    """
    print("\nGenerating Comparative Gains Plot...")
    
    # Prepare Data
    norag_df = results_df[results_df['Scenario'] == "No RAG (LLM Only)"].set_index('Query')
    
    # Identify No-RAG Hallucinations (Score <= 2.0)
    norag_hallucinations = norag_df[norag_df['Faithfulness'] <= 2.0].index
    
    metrics = []
    
    for scenario in ["RAG + Abstain", "RAG (No Abstain)"]:
        if scenario not in results_df['Scenario'].unique():
            continue
            
        rag_df = results_df[results_df['Scenario'] == scenario].set_index('Query')
        
        # 1. Hallucinations Fixed (Count)
        # Count how many of the 'norag_hallucinations' have score > 2.0 in RAG
        fixed_count = 0
        if len(norag_hallucinations) > 0:
            for q in norag_hallucinations:
                if q in rag_df.index and rag_df.loc[q, 'Faithfulness'] > 2.0:
                    fixed_count += 1
            
        # 2. Queries Improved (Count)
        # Count queries where RAG > No RAG
        common_queries = rag_df.index.intersection(norag_df.index)
        improved_count = 0
        for q in common_queries:
            if rag_df.loc[q, 'Faithfulness'] > norag_df.loc[q, 'Faithfulness']:
                improved_count += 1
        
        metrics.append({
            "Scenario": scenario,
            "Metric": "Hallucinations Fixed (Count)",
            "Value": fixed_count
        })
        metrics.append({
            "Scenario": scenario,
            "Metric": "Queries Improved (Count)",
            "Value": improved_count
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    if metrics_df.empty:
        print("No comparative data available.")
        return

    plt.figure(figsize=(10, 6))
    # Removed local style override to maintain global ggplot style
    
    # Custom palette
    palette = {"RAG + Abstain": "#90CAF9", "RAG (No Abstain)": "#42A5F5"}
    
    ax = sns.barplot(
        data=metrics_df,
        x="Metric",
        y="Value",
        hue="Scenario",
        palette=palette
    )
    
    total_queries = len(results_df['Query'].unique())
    plt.title(f"RAG vs No-RAG: Comparative Improvements (Total Queries: {total_queries})", fontsize=14)
    plt.ylabel("Count", fontsize=12)
    plt.xlabel("")
    
    # Add labels (integers)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3, fontweight='bold')
        
    plt.legend(title="Scenario", loc='upper right')
    plt.tight_layout()
    
    if save_path:
        _save_plot(save_path, 'comparative_gains.png')
        
    plt.show()