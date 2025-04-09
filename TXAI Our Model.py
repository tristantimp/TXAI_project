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

src_lang = "nl_XX"
tgt_lang = "en_XX"
tokenizer.src_lang = src_lang

# === Mini-datasets ===
datasets = {
    "MT": [
        ("Ik hou van programmeren.", "I love programming."),
        ("Het regent vandaag.", "It is raining today."),
        ("De kat zit op de mat.", "The cat is sitting on the mat.")
    ],
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
}

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
    "TopKVar": topk_variance,
    "NormEntropy": normalized_entropy,
}

# === Evaluatie loop ===
rows = []

for task_name, samples in datasets.items():
    print(f"\n🔍 Evaluating task: {task_name}")
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
            tokens = output.sequences[0]
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            logits = torch.stack(output.scores).squeeze(1)

        # Bereken ROUGE & BERTScore
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = rouge.score(reference, generated_text)['rougeL'].fmeasure
        _, _, bert = bert_score([generated_text], [reference], lang="en", verbose=False)
        bert = bert.numpy().item()

        # UE scores
        for method_name, func in ue_methods.items():
            ue_score = func(logits)
            rows.append({
                "Task": task_name,
                "Method": method_name,
                "UE_Score": ue_score,
                "ROUGE-L": rouge_l,
                "BERTScore": bert
            })

# === Tabel maken ===
df = pd.DataFrame(rows)
summary = df.groupby(["Task", "Method"])[["ROUGE-L", "BERTScore"]].mean().round(3)

# === Heatmap visualiseren ===
plt.figure(figsize=(10, 6))
sns.heatmap(summary, annot=True, cmap="Greens", fmt=".2f", linewidths=0.5, cbar=True)
plt.title("Uncertainty Estimation Performance (donkergroen = beter)", fontsize=14)
plt.tight_layout()

# === Opslaan als afbeelding ===
plt.savefig("ue_table_groen_heatmap.png")
plt.show()
