import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
from plotting import ue_table, plot_PRR

# Setup 
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

src_lang = "en_XX"
tgt_lang = "nl_XX"
tokenizer.src_lang = src_lang

# Dataset
datasets = {
    "MT": [
        ("I love programming.", "Ik hou van programmeren."),
        ("It is raining today.", "Het regent vandaag."),
        ("Will you invite me to your brother-in-law's birthday?.", "Nodig je mij uit naar je zwager's verjaardag?."),
        ("It is raining cats and dogs.", "Het regent pijpestelen."),
        ("I disagree.", "Ik ben het er niet mee eens.")
    ]} 


# UE Methods
def mean_token_entropy(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropies = -(probs * log_probs).sum(dim=-1)
    return entropies.mean().item()

def max_sequence_probability(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1).values
    return max_probs.mean().item()

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

# Evaluation loop
rows = []

for task_name, samples in datasets.items():
    print(f"Evaluating task: {task_name}")
    for input_text, reference in tqdm(samples):
        # Input
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
        inputs['forced_bos_token_id'] = tokenizer.lang_code_to_id[tgt_lang]

        # Generate output
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                return_dict_in_generate=True,
                output_scores=True
            )
            tokens = output.sequences[0]
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            print("\n generated text ", generated_text)
            logits = torch.stack(output.scores).squeeze(1)
            logits = torch.clamp(logits, min=-1e9, max=1e9)

        # Calculate ROUGE & BERTScore
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = rouge.score(reference, generated_text)['rougeL'].fmeasure
        _, _, bert = bert_score([generated_text], [reference], lang="nl", verbose=False)
        bert = bert.numpy().item()

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

df = pd.DataFrame(rows)
print("\n\n dataframe: ", df)

ue_table(df, rows)

# Sort predictions by uncertainty score
df_sorted = df.sort_values(by="UE_Score", ascending=True)

plot_PRR(df_sorted)
