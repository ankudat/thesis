import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. DIRECTORY CONFIGURATION
# Using dynamic path lookup to ensure the script works relative to its folder
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

# Ensure the plots subfolder exists
os.makedirs(PLOT_DIR, exist_ok=True)

# Path mapping to your specific JSON result files
FILE_MAP = {
    # Classical baselines
    'BERT (FT)':      os.path.join(RESULTS_DIR, "bert_finetuned", "bert_finetuned_evaluation_results.json"),
    'BERT (Base)':    os.path.join(RESULTS_DIR, "classical_baselines", "bert", "bert_evaluation_results.json"),
    'SPACY':          os.path.join(RESULTS_DIR, "classical_baselines", "spacy", "spacy_evaluation_results.json"),
    'MS Presidio':    os.path.join(RESULTS_DIR, "presidio_baseline", "presidio_evaluation_results.json"),
    # LLM baselines: Llama-3
    'Llama3 (FS+V)':  os.path.join(RESULTS_DIR, "llm_baselines", "llm_meta_llama_3_8b_instruct_few_shot_verified_evaluation_results.json"),
    'Llama3 (FS)':    os.path.join(RESULTS_DIR, "llm_baselines", "llm_meta_llama_3_8b_instruct_few_shot_evaluation_results.json"),
    'Llama3 (ZS)':    os.path.join(RESULTS_DIR, "llm_baselines", "llm_meta_llama_3_8b_instruct_zero_shot_evaluation_results.json"),
    # LLM baselines: Qwen2.5
    'Qwen2.5 (FS+V)': os.path.join(RESULTS_DIR, "llm_baselines", "llm_qwen2.5_7b_instruct_few_shot_verified_evaluation_results.json"),
    # LLM baselines: SauerkrautLM
    'SauerkrautLM (FS+V)': os.path.join(RESULTS_DIR, "llm_baselines", "llm_llama_3.1_sauerkrautlm_8b_instruct_few_shot_verified_evaluation_results.json"),
    # LLM fine-tuned (QLoRA)
    'Llama3 (FT)':    os.path.join(RESULTS_DIR, "llm_finetuned", "llm_finetuned_meta_llama_3_8b_instruct", "llm_finetuned_meta_llama_3_8b_instruct_evaluation_results.json"),
    'Qwen2.5 (FT)':   os.path.join(RESULTS_DIR, "llm_finetuned", "llm_finetuned_qwen2.5_7b_instruct", "llm_finetuned_qwen2.5_7b_instruct_evaluation_results.json"),
    'SauerkrautLM (FT)': os.path.join(RESULTS_DIR, "llm_finetuned", "llm_finetuned_llama_3.1_sauerkrautlm_8b_instruct", "llm_finetuned_llama_3.1_sauerkrautlm_8b_instruct_evaluation_results.json"),
}

# 2. ACADEMIC COLOR PALETTE
# Blues for LLMs (one shade per model family), Grays for classical baselines
# Fine-tuned variants use darker/saturated versions of their family color
BLUE_GRAY_PALETTE = {
    # Classical baselines (grays)
    'BERT (FT)':      '#4d4d4d',   # Dark Gray
    'BERT (Base)':    '#7f7f7f',   # Gray
    'SPACY':          '#afafaf',   # Silver
    'MS Presidio':    '#d9d9d9',   # Light Gray
    # Llama-3 (blues)
    'Llama3 (FS+V)':  '#2171b5',   # Medium Blue
    'Llama3 (FS)':    '#6baed6',   # Light Blue
    'Llama3 (ZS)':    '#9ecae1',   # Very Light Blue
    'Llama3 (FT)':    '#08306b',   # Deep Navy (fine-tuned = darkest)
    # Qwen2.5 (greens)
    'Qwen2.5 (FS+V)': '#41ab5d',  # Medium Green
    'Qwen2.5 (FT)':   '#00441b',  # Deep Green (fine-tuned = darkest)
    # SauerkrautLM (oranges)
    'SauerkrautLM (FS+V)': '#f16913',  # Medium Orange
    'SauerkrautLM (FT)':   '#8c2d04',  # Deep Burnt Orange (fine-tuned = darkest)
}

# 3. DATA PROCESSING
def load_data():
    rows = []
    for model_name, path in FILE_MAP.items():
        if not os.path.exists(path):
            print(f"Warning: Missing file for {model_name} at {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            res = json.load(f)
            rows.append({
                'Model': model_name,
                'Overall': res['Overall']['strict']['All Categories']['overall']['f1'],
                'Tier 1': res['Overall']['strict']['Tier 1 – Direct NER']['overall']['f1'],
                'Tier 2': res['Overall']['strict']['Tier 2 – Structured (Regex)']['overall']['f1'],
                'Tier 3': res['Overall']['strict']['Tier 3 – Quasi-Identifiers']['overall']['f1'],
                'Low': res['Low']['strict']['All Categories']['overall']['f1'],
                'Medium': res['Medium']['strict']['All Categories']['overall']['f1'],
                'High': res['High']['strict']['All Categories']['overall']['f1'],
            })
    return pd.DataFrame(rows)

df = load_data()
ranked_models = df.sort_values('Overall', ascending=False)['Model'].tolist()

# 4. GLOBAL PLOT STYLING (Font: Arial)
sns.set_theme(style="ticks")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white'
})

def finalize_and_save(fig, filename):
    plt.tight_layout()
    full_path = os.path.join(PLOT_DIR, filename)
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {full_path}")
    plt.close(fig)

# --- CHART 1: OVERALL PERFORMANCE ---
fig1, ax1 = plt.subplots(figsize=(10, 7))
df_sorted = df.sort_values('Overall', ascending=False)
sns.barplot(x='Overall', y='Model', data=df_sorted, palette=BLUE_GRAY_PALETTE, edgecolor='black', ax=ax1)
ax1.set_xlim(0.4, 1.05)
for i, v in enumerate(df_sorted['Overall']):
    ax1.text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold')
ax1.set_title('Overall NER Performance Ranking (Strict $F_{1}$ Score)')
ax1.set_xlabel('$F_{1}$ Score')
ax1.set_ylabel('')
sns.despine()
finalize_and_save(fig1, 'overall_performance.png')

# --- CHART 2: TIER COMPARISON ---
tier_df = df.melt(id_vars='Model', value_vars=['Tier 1', 'Tier 2', 'Tier 3'], var_name='Tier', value_name='F1')
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(data=tier_df, x='Tier', y='F1', hue='Model', hue_order=ranked_models, palette=BLUE_GRAY_PALETTE, edgecolor='black', ax=ax2)
ax2.set_ylim(0, 1.15)
ax2.legend(title='Methods', bbox_to_anchor=(1.02, 1), loc='upper left')
ax2.set_title('Performance Comparison by Entity Tier')
ax2.set_ylabel('$F_{1}$ Score')
ax2.set_xlabel('')
sns.despine()
finalize_and_save(fig2, 'tier_comparison.png')

# --- CHART 3: COMPLEXITY ROBUSTNESS ---
comp_df = df.melt(id_vars='Model', value_vars=['Low', 'Medium', 'High'], var_name='Complexity', value_name='F1')
fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.barplot(data=comp_df, x='Complexity', y='F1', hue='Model', hue_order=ranked_models, palette=BLUE_GRAY_PALETTE, edgecolor='black', ax=ax3)
ax3.set_ylim(0.4, 1.15)
ax3.legend(title='Methods', bbox_to_anchor=(1.02, 1), loc='upper left')
ax3.set_title('Model Robustness by Document Complexity Level')
ax3.set_ylabel('$F_{1}$ Score')
ax3.set_xlabel('')
sns.despine()
finalize_and_save(fig3, 'complexity_robustness.png')