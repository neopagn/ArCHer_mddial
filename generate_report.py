import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

def load_experiment_results(results_path):
    """Load experiment results from CSV"""
    return pd.read_csv(results_path)

def generate_metrics_summary(results):
    """Generate summary statistics for metrics"""
    summary = {
        'accuracy': {
            'mean': results['accuracy'].mean(),
            'std': results['accuracy'].std(),
            'min': results['accuracy'].min(),
            'max': results['accuracy'].max()
        },
        'avg_reward': {
            'mean': results['avg_reward'].mean(),
            'std': results['avg_reward'].std(),
            'min': results['avg_reward'].min(),
            'max': results['avg_reward'].max()
        },
        'avg_questions': {
            'mean': results['avg_questions'].mean(),
            'std': results['avg_questions'].std(),
            'min': results['avg_questions'].min(),
            'max': results['avg_questions'].max()
        },
        'avg_time': {
            'mean': results['avg_time'].mean(),
            'std': results['avg_time'].std(),
            'min': results['avg_time'].min(),
            'max': results['avg_time'].max()
        }
    }
    return summary

def create_visualizations(results, output_dir):
    """Create visualizations for the report"""
    # Set style
    plt.style.use('seaborn')
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results, x='model', y='accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 2. Reward Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results, x='model', y='avg_reward')
    plt.title('Reward Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_distribution.png'))
    plt.close()
    
    # 3. Questions vs Time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results, x='avg_questions', y='avg_time', hue='model')
    plt.title('Questions vs Time')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'questions_vs_time.png'))
    plt.close()

def generate_latex_report(results, summary, output_dir):
    """Generate LaTeX report"""
    report = f"""\\documentclass{{article}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{float}}
\\usepackage{{hyperref}}

\\title{{Medical Diagnosis Dialogue System: Final Report}}
\\author{{Your Name}}
\\date{{{datetime.now().strftime('%B %d, %Y')}}}

\\begin{{document}}
\\maketitle

\\section{{Introduction}}
This report presents the results of our medical diagnosis dialogue system, which uses a T5-based model trained with behavioral cloning and reinforcement learning.

\\section{{System Architecture}}
Our system consists of:
\\begin{{itemize}}
    \\item MDDial Environment for simulating doctor-patient dialogues
    \\item T5-based model for generating questions and diagnoses
    \\item Reward system for evaluating model performance
\\end{{itemize}}

\\section{{Experimental Results}}

\\subsection{{Metrics Summary}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lrrrr}}
\\toprule
Metric & Mean & Std & Min & Max \\\\
\\midrule
Accuracy & {summary['accuracy']['mean']:.3f} & {summary['accuracy']['std']:.3f} & {summary['accuracy']['min']:.3f} & {summary['accuracy']['max']:.3f} \\\\
Average Reward & {summary['avg_reward']['mean']:.3f} & {summary['avg_reward']['std']:.3f} & {summary['avg_reward']['min']:.3f} & {summary['avg_reward']['max']:.3f} \\\\
Average Questions & {summary['avg_questions']['mean']:.1f} & {summary['avg_questions']['std']:.1f} & {summary['avg_questions']['min']:.1f} & {summary['avg_questions']['max']:.1f} \\\\
Average Time (s) & {summary['avg_time']['mean']:.1f} & {summary['avg_time']['std']:.1f} & {summary['avg_time']['min']:.1f} & {summary['avg_time']['max']:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Summary of Model Performance Metrics}}
\\end{{table}}

\\subsection{{Visualizations}}
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{os.path.join(output_dir, 'accuracy_comparison.png')}}}
\\caption{{Model Accuracy Comparison}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{os.path.join(output_dir, 'reward_distribution.png')}}}
\\caption{{Reward Distribution}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{os.path.join(output_dir, 'questions_vs_time.png')}}}
\\caption{{Questions vs Time Analysis}}
\\end{{figure}}

\\section{{Discussion}}
Our model achieves an accuracy of {summary['accuracy']['mean']:.3f}, which is comparable to the baseline model. The system takes an average of {summary['avg_questions']['mean']:.1f} questions to reach a diagnosis, with an average time of {summary['avg_time']['mean']:.1f} seconds per case.

\\section{{Conclusion}}
The results demonstrate the effectiveness of our approach in automating medical diagnosis through dialogue. Future work could focus on:
\\begin{{itemize}}
    \\item Expanding the disease coverage
    \\item Improving question generation quality
    \\item Reducing diagnosis time
\\end{{itemize}}

\\end{{document}}
"""
    
    with open(os.path.join(output_dir, 'report.tex'), 'w') as f:
        f.write(report)

def main():
    # Create output directory
    output_dir = 'report_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_experiment_results('model_comparison.csv')
    
    # Generate summary
    summary = generate_metrics_summary(results)
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Generate LaTeX report
    generate_latex_report(results, summary, output_dir)
    
    print(f"Report generated in {output_dir}/")
    print("To compile the report, run: pdflatex report.tex")

if __name__ == "__main__":
    main() 