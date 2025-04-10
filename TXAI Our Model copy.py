import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

src_lang = "en_XX"
tgt_lang = "nl_XX"
tokenizer.src_lang = src_lang

# === Mini-datasets ===
datasets = {
    "MT": [
        ("I love programming.", "Ik hou van programmeren."),
        ("It is raining today.", "Het regent vandaag."),
        ("Will you invite me to your brother-in-law's birthday?.", "Nodig je mij uit naar je zwager's verjaardag?."),
        ("It is raining cats and dogs.", "Het regent pijpestelen."),
        ("I disagree.", "Ik ben het er niet mee eens.")

    ]} 
'''
    "TS": [
        ("De overheid heeft vandaag aangekondigd dat de belastingen volgend jaar zullen stijgen vanwege economische omstandigheden.", "Belastingen stijgen volgend jaar."),
        ("Na een spannende wedstrijd won Ajax met 2-1 van PSV in de laatste minuut.", "Ajax wint met 2-1 van PSV."),
        ("Wetenschappers ontdekten een nieuwe planeet buiten ons zonnestelsel.", "Nieuwe planeet ontdekt.")
    ],
    "QA": [
        ("Wat is de hoofdstad van Nederland? Amsterdam is de grootste stad van Nederland en ook de hoofdstad.", "Amsterdam"),
        ("Wie schreef het boek 'De Avonden'? Gerard Reve schreef het in 1947.", "Gerard Reve"),
        ("Wat is de langste rivier in Nederland? De Rijn stroomt door meerdere landen en is de langste rivier.", "De Rijn")
    ]
}'''

# === UE Methoden ===
def mean_token_entropy(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropies = -(probs * log_probs).sum(dim=-1)
    return entropies.mean().item()

def max_sequence_probability(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1).values
    return max_probs.mean().item()

def topk_variance(logits, k=5):
    topk_vals = torch.topk(logits, k=k, dim=-1).values
    return topk_vals.var(dim=-1).mean().item()

def normalized_entropy(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_entropy = np.log(logits.shape[-1])
    return (entropy / max_entropy).mean().item()

ue_methods = {
    "MeanEntropy": mean_token_entropy,
    "MaxSeqProb": max_sequence_probability,
    "NormEntropy": normalized_entropy,
}

# === Evaluatie loop ===
rows = []

for task_name, samples in datasets.items():
    print(f"Evaluating task: {task_name}")
    for input_text, reference in tqdm(samples):
        # Invoeren
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
        inputs['forced_bos_token_id'] = tokenizer.lang_code_to_id[tgt_lang]

        # Genereer output
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                return_dict_in_generate=True,
                output_scores=True
            )
            #print("output", output)
            tokens = output.sequences[0]
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            print("\n generated text ", generated_text)
            logits = torch.stack(output.scores).squeeze(1)
            logits = torch.clamp(logits, min=-1e9, max=1e9)
            #print("\n logits", logits.shape)

        # Bereken ROUGE & BERTScore
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = rouge.score(reference, generated_text)['rougeL'].fmeasure
        _, _, bert = bert_score([generated_text], [reference], lang="nl", verbose=False)
        bert = bert.numpy().item()
        #print("\n rouge l and bert",rouge_l, bert)

        # UE scores
        for method_name, func in ue_methods.items():
            ue_score = func(logits)
            print(f"{method_name} score: {ue_score:.4f}")
            rows.append({
                "Task": task_name,
                "Method": method_name,
                "UE_Score": ue_score,
                "ROUGE-L": rouge_l,
                "BERTScore": bert
            })

# === Tabel maken ===
df = pd.DataFrame(rows)
print("\n\n dataframe: ", df)
#summary = df.groupby(["Task", "Method"])[["ROUGE-L", "BERTScore"]].mean().round(3)
#print("\n\n summary: ", summary)

# Pivot the DataFrame to prepare for heatmap
heatmap_data = df.pivot_table(
    index=["Task", "Method"], 
    values=["UE_Score", "ROUGE-L", "BERTScore"], 
    aggfunc="mean"
)

# Adjust the data for plotting
heatmap_data = heatmap_data.reset_index().set_index(["Task", "Method"])
heatmap_data = heatmap_data.stack().reset_index(name="Value")
heatmap_data.columns = ["Task", "Method", "Metric", "Value"]

# Create a pivot table for heatmap
pivot_table = heatmap_data.pivot_table(
    index=["Task", "Method"], 
    columns="Metric", 
    values="Value"
)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_table, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu", 
    linewidths=0.5, 
    cbar=True
)
plt.title("Heatmap of Metrics by Task and Method", fontsize=14)
plt.xlabel("Metric", fontsize=12)
plt.ylabel("Task and Method", fontsize=12)
plt.tight_layout()

# Save the heatmap
plt.savefig("ue_metrics_heatmap.png")
plt.show()


from sklearn.metrics import auc

def compute_auc(pr_curve):
    x = [i / len(pr_curve) for i in range(len(pr_curve))]
    return auc(x, pr_curve)

# === Compute UE Scores and Add to DataFrame ===
def compute_ue_scores(df, ue_methods):
    for method_name, func in ue_methods.items():
        df[method_name] = df["logits"].apply(func)
    return df

# Apply UE calculations to the DataFrame
df = compute_ue_scores(df, ue_methods)

# Step 2: Compute PR curves for UE methods and quality metrics
def compute_pr_curve_for_ue(df_sorted, quality_metric, ue_method):
    pr_curve = []
    n = len(df_sorted)
    
    for rejection_rate in range(n):
        # Retain predictions with lowest uncertainty scores based on the selected UE method
        retained = df_sorted.iloc[:n - rejection_rate]
        
        # Compute average quality over retained predictions
        avg_quality = retained[quality_metric].mean()
        pr_curve.append(avg_quality)
    
    return pr_curve

# Compute PRR for each UE method and quality metric (ROUGE-L and BERTScore)
prr_results = {}

for method_name in ue_methods:
    # Sort by the specific UE method
    df_sorted = df.sort_values(by=method_name, ascending=True)

    # PR curves for ROUGE-L and BERTScore
    pr_rouge = compute_pr_curve_for_ue(df_sorted, "ROUGE-L", method_name)
    pr_bert = compute_pr_curve_for_ue(df_sorted, "BERTScore", method_name)

    # Compute areas under PR curves
    aucpr_unc_rouge = compute_auc(pr_rouge)
    aucpr_unc_bert = compute_auc(pr_bert)

    # Random rejection curve (baseline)
    random_curve = [df_sorted["ROUGE-L"].mean()] * (len(df_sorted) + 1)
    aucpr_random_rouge = compute_auc(random_curve)

    # Oracle rejection curve (ideal uncertainty estimation)
    oracle_curve_rouge = sorted(df_sorted["ROUGE-L"], reverse=True)
    aucpr_oracle_rouge = compute_auc(oracle_curve_rouge)

    # Compute PRR for ROUGE-L
    prr_rouge = (aucpr_unc_rouge - aucpr_random_rouge) / (aucpr_oracle_rouge - aucpr_random_rouge)

    # Random rejection curve (baseline) for BERTScore
    random_curve_bert = [df_sorted["BERTScore"].mean()] * (len(df_sorted) + 1)
    aucpr_random_bert = compute_auc(random_curve_bert)

    # Oracle rejection curve (ideal uncertainty estimation) for BERTScore
    oracle_curve_bert = sorted(df_sorted["BERTScore"], reverse=True)
    aucpr_oracle_bert = compute_auc(oracle_curve_bert)

    # Compute PRR for BERTScore
    prr_bert = (aucpr_unc_bert - aucpr_random_bert) / (aucpr_oracle_bert - aucpr_random_bert)

    # Store results for each UE method
    prr_results[method_name] = {
        "PRR (ROUGE-L)": prr_rouge,
        "PRR (BERTScore)": prr_bert
    }

# Display PRR values
for method_name, scores in prr_results.items():
    print(f"{method_name} - PRR (ROUGE-L): {scores['PRR (ROUGE-L)']:.4f}")
    print(f"{method_name} - PRR (BERTScore): {scores['PRR (BERTScore)']:.4f}")
