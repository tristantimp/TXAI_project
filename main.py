import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
from plotting import *
from UE_methods import *

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
            #print("\n generated text ", generated_text)
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
            #print(f"{method_name} score: {ue_score:.4f}")
            rows.append({
                "Task": task_name,
                "Method": method_name,
                "UE Score": ue_score,
                "ROUGE-L": rouge_l,
                "BERTScore": bert
            })
            
# create dafaframe of uncertainty estimates
df = pd.DataFrame(rows)

# group uncertainty estimates by UE method
ue_grouped_table(df)

# create heatmap of uncertainty estimates
ue_heatmap(df, rows)

# separate and sort dataframe by method
df1, df2, df3 = separate_df(df)

# compute and plot PRR for each UE method
prr_rouge, prr_bert = plot_PRR(df1, "MaxSeqProb")
prr_rouge2, prr_bert2 = plot_PRR(df2, "MeanEntropy") 
prr_rouge3, prr_bert3 = plot_PRR(df3, "NormEntropy")  

PRR_table(prr_rouge, prr_rouge2, prr_rouge3, prr_bert, prr_bert2, prr_bert3)

