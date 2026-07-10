import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_f1_bar():
    # Authoritative numbers (N=600, bootstrap B=1000 seed 42)
    # Pure Vector: 0.663
    # Pure Graph: 0.633
    # Pure Hybrid: 0.647
    # Single-stage: 0.457
    # Two-stage (selective): 0.602
    # Always-on: 0.682
    # Oracle: 0.524
    # Oracle+Stage-2: 0.615
    
    systems = ['Oracle', 'Oracle+S2', 'Single-stage', 'Two-stage', 'Always-on', 'Pure Hybrid', 'Pure Graph', 'Pure Vector']
    f1_scores = [0.524, 0.615, 0.457, 0.602, 0.682, 0.647, 0.633, 0.663]
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Custom colors: highlight Always-on, Oracle+S2, and Pure Vector
    colors = ['#cccccc', '#8c564b', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#17becf']
    
    bars = plt.barh(systems, f1_scores, color=colors)
    
    plt.xlabel('Token-level F1 Score')
    plt.title('End-to-End Answer Quality by System')
    plt.xlim(0, 0.8)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', ha='left', va='center', fontweight='bold')
                 
    plt.tight_layout()
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/f1_bar.pdf', format='pdf', dpi=1200)
    plt.savefig('figs/f1_bar.eps', format='eps', dpi=1200)
    print("Saved f1_bar.pdf and f1_bar.eps")

def plot_dataset_bar():
    classes = ['Dense Retrieval', 'Graph Traversal', 'Hybrid Reasoning']
    counts = [300, 150, 150]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, counts, color=['#4C72B0', '#DD8452', '#55A868'], edgecolor='black', width=0.6)
    
    plt.ylabel('Number of Queries', fontsize=12)
    plt.title('Evaluation Dataset Distribution', fontsize=14)
    plt.ylim(0, 350)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{int(yval)}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('figs/dataset_bar.pdf', format='pdf', dpi=1200)
    plt.savefig('figs/dataset_bar.eps', format='eps', dpi=1200)
    print("Saved dataset_bar.pdf and dataset_bar.eps")

if __name__ == '__main__':
    plot_f1_bar()
    plot_dataset_bar()
