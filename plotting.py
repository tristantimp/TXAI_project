import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc
import pandas as pd

# plot the heatmap of UE scores
def ue_heatmap(df, rows):
    # Pivot the DataFrame to prepare for heatmap
    heatmap_data = df.pivot_table(
        index=["Task", "Method"], 
        values=["UE Score", "ROUGE-L", "BERTScore"], 
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

    plt.savefig("figures/ue_metrics_heatmap.png")
    plt.show()

def compute_pr_curve(df_sorted, quality_metric):
    pr_curve = []
    n = len(df_sorted)
    
    for rejection_rate in range(n):
        # Retain predictions with lowest uncertainty scores
        retained = df_sorted.iloc[:n - rejection_rate]
        
        # Compute average quality over retained predictions
        avg_quality = retained[quality_metric].mean()
        pr_curve.append(avg_quality)
    
    return pr_curve

def compute_auc(pr_curve):
    x = [i / len(pr_curve) for i in range(len(pr_curve))]
    return auc(x, pr_curve)

def plot_PRR(df_sorted, title):
    pr_rouge = compute_pr_curve(df_sorted, "ROUGE-L")
    pr_bert = compute_pr_curve(df_sorted, "BERTScore")

    aucpr_unc_rouge = compute_auc(pr_rouge)
    aucpr_unc_bert = compute_auc(pr_bert)

    # Random rejection curve (baseline)
    random_curve = [df_sorted["ROUGE-L"].mean()] * (len(df_sorted) + 1)
    aucpr_random_rouge = compute_auc(random_curve)

    # Oracle rejection curve (ideal uncertainty estimation)
    oracle_curve_rouge = sorted(df_sorted["ROUGE-L"], reverse=True)
    aucpr_oracle_rouge = compute_auc(oracle_curve_rouge)

    # Compute PRR
    prr_rouge = (aucpr_unc_rouge - aucpr_random_rouge) / (aucpr_oracle_rouge - aucpr_random_rouge)

    # Repeat for BERTScore
    random_curve_bert = [df_sorted["BERTScore"].mean()] * (len(df_sorted) + 1)
    aucpr_random_bert = compute_auc(random_curve_bert)

    oracle_curve_bert = sorted(df_sorted["BERTScore"], reverse=True)
    aucpr_oracle_bert = compute_auc(oracle_curve_bert)

    prr_bert = (aucpr_unc_bert - aucpr_random_bert) / (aucpr_oracle_bert - aucpr_random_bert)

    plt.figure(figsize=(10, 6))
    plt.plot(pr_rouge, label="PR Curve (ROUGE-L)", marker='o')
    plt.plot(pr_bert, label="PR Curve (BERTScore)", marker='o')
    plt.axhline(y=random_curve[0], color='gray', linestyle='--', label="Random")
    plt.title("Prediction Rejection (PR) Curves for " + title)
    plt.xlabel("Rejection Rate")
    plt.ylabel("Quality Metric (Q)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/pr_curves_" + title +".png")
    plt.show()

    return prr_rouge, prr_bert

# create a table of UE scores
def ue_grouped_table(df):
    grouped_df = df.groupby('Method').apply(lambda x: x).reset_index(drop=True)
    grouped_df = grouped_df.round(3)
    grouped_df.to_csv("figures/ue_scores_table.csv", index=False)

#separate and sort the dataframe by UE method
def separate_df(df):
    grouped_df = df.groupby('Method').apply(lambda x: x).reset_index(drop=True)
    df1 = grouped_df[:10]
    df2 = grouped_df[10:20]
    df3 = grouped_df[20:]
    df1 = df1.sort_values(by="UE Score", ascending=True)
    df2 = df2.sort_values(by="UE Score", ascending=True)
    df3 = df3.sort_values(by="UE Score", ascending=True)
    return df1, df2, df3

#create a table of PRR scores
def PRR_table(prr_rouge,prr_rouge2,prr_rouge3,prr_bert, prr_bert2, prr_bert3):
    data = {"ROUGE-L": [prr_rouge, prr_rouge2, prr_rouge3],
    "BERTScore": [prr_bert, prr_bert2, prr_bert3]}
    index = ["MaxSeqProb", "MeanEntropy", "NormEntropy"]
    df = pd.DataFrame(data, index=index)
    df.to_csv("figures/PRR_table.csv")
