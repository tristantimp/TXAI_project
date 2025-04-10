import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc

def ue_table(df, rows):
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

def plot_PRR(df_sorted):
    pr_rouge = compute_pr_curve(df_sorted, "ROUGE-L")
    print("\n\n pr_rouge", pr_rouge)
    pr_bert = compute_pr_curve(df_sorted, "BERTScore")
    print("\n\n pr_bert", pr_bert)

    aucpr_unc_rouge = compute_auc(pr_rouge)
    print("\n\n aucpr_unc_rouge", aucpr_unc_rouge)
    aucpr_unc_bert = compute_auc(pr_bert)
    print("\n\n aucpr_unc_bert", aucpr_unc_bert)

    # Random rejection curve (baseline)
    random_curve = [df_sorted["ROUGE-L"].mean()] * (len(df_sorted) + 1)
    aucpr_random_rouge = compute_auc(random_curve)
    print("\n\n aucpr_random_rouge", aucpr_random_rouge)

    # Oracle rejection curve (ideal uncertainty estimation)
    oracle_curve_rouge = sorted(df_sorted["ROUGE-L"], reverse=True)
    aucpr_oracle_rouge = compute_auc(oracle_curve_rouge)
    print("\n\n aucpr oracle rouge", aucpr_oracle_rouge)

    # Step 4: Compute PRR
    prr_rouge = (aucpr_unc_rouge - aucpr_random_rouge) / (aucpr_oracle_rouge - aucpr_random_rouge)
    #print("prr rouge", prr_rouge)

    # Repeat for BERTScore
    random_curve_bert = [df_sorted["BERTScore"].mean()] * (len(df_sorted) + 1)
    aucpr_random_bert = compute_auc(random_curve_bert)
    print("\n\n aucpr_random_bert", aucpr_random_bert)

    oracle_curve_bert = sorted(df_sorted["BERTScore"], reverse=True)
    aucpr_oracle_bert = compute_auc(oracle_curve_bert)
    print("\n\n aucpr oracle bert", aucpr_oracle_bert)

    prr_bert = (aucpr_unc_bert - aucpr_random_bert) / (aucpr_oracle_bert - aucpr_random_bert)
    #print("prr bert", prr_bert)

    # Step 5: Plot PR curves
    plt.figure(figsize=(10, 6))
    plt.plot(pr_rouge, label="PR Curve (ROUGE-L)", marker='o')
    plt.plot(pr_bert, label="PR Curve (BERTScore)", marker='o')
    plt.axhline(y=random_curve[0], color='gray', linestyle='--', label="Random")
    plt.title("Prediction Rejection (PR) Curves")
    plt.xlabel("Rejection Rate")
    plt.ylabel("Quality Metric (Q)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/pr_curves.png")
    plt.show()

    # Display PRR values
    print(f"PRR (ROUGE-L): {prr_rouge:.4f}")
    print(f"PRR (BERTScore): {prr_bert:.4f}")

