import matplotlib.pyplot as plt
import numpy as np

def plot_scientific_validation(results, save_path="validation.png"):
    """Cellule 6 : Scatter Plot Q vs NMI"""
    modularity_scores = [r[1] for r in results]
    nmi_scores = [r[2] for r in results]

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)
    plt.scatter(modularity_scores, nmi_scores, color='blue', alpha=0.7, edgecolors='k')
    
    # Trend line
    if len(results) > 1:
        z = np.polyfit(modularity_scores, nmi_scores, 1)
        p = np.poly1d(z)
        plt.plot(modularity_scores, p(modularity_scores), "r--", alpha=0.5)

    plt.xlabel('Modularity (Q)')
    plt.ylabel('NMI Accuracy')
    plt.title('Validation of Evolutionary Selection Strategy')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")

def plot_benchmark_comparison(my_score, save_path="benchmark.png"):
    """Cellule 7 : Benchmark vs Louvain/Node2Vec"""
    benchmarks = {'Louvain': 0.4300, 'Node2Vec': 0.4700, 'FE-ACO (Yours)': my_score}
    names = list(benchmarks.keys())
    scores = list(benchmarks.values())
    colors = ['gray', 'royalblue', '#00cc00']
    
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)
    bars = plt.bar(names, scores, color=colors, width=0.6)
    plt.axhline(y=0.43, color='red', linestyle='--')
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.4f}', ha='center')
    
    plt.title('State-of-the-Art Comparison')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Benchmark saved: {save_path}")