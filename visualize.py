"""
visualize.py
------------
Generates exploratory visualizations from heart.csv and saves them as
PNG files in the 'visuals' folder.

Charts produced:
    01_thal_class_distribution.png   - bar chart of thal class counts
    02_correlation_heatmap.png       - correlation matrix of all numeric features
    03_age_distribution.png          - histogram of age, split by thal
    04_thalach_by_thal.png           - boxplot of max heart rate by thal class
    05_chol_by_thal.png              - boxplot of cholesterol by thal class
    06_target_vs_thal.png            - stacked bar of heart disease target vs thal
    07_pairwise_scatter.png          - scatter matrix of key numeric features

Run:
    python visualize.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "heart.csv"
OUT_DIR = "visuals"
os.makedirs(OUT_DIR, exist_ok=True)

THAL_LABELS = {0: "unknown", 1: "normal", 2: "fixed defect", 3: "reversable defect"}

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()
print(f"Loaded {len(df)} rows from '{DATA_PATH}' after removing duplicates")

df["thal_label"] = df["thal"].map(THAL_LABELS)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------
# 2. Thal class distribution
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
order = sorted(df["thal_label"].dropna().unique(), key=lambda l: list(THAL_LABELS.values()).index(l))
sns.countplot(data=df, x="thal_label", order=order, hue="thal_label",
              palette="viridis", legend=False, ax=ax)
ax.set_title("Distribution of 'thal' Classes")
ax.set_xlabel("thal")
ax.set_ylabel("Count")
for container in ax.containers:
    ax.bar_label(container)
savefig(fig, "01_thal_class_distribution.png")

# ---------------------------------------------------------------------
# 3. Correlation heatmap (numeric features only)
# ---------------------------------------------------------------------
numeric_df = df.select_dtypes(include="number")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=ax)
ax.set_title("Correlation Heatmap of Numeric Features")
savefig(fig, "02_correlation_heatmap.png")

# ---------------------------------------------------------------------
# 4. Age distribution split by thal
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x="age", hue="thal_label", multiple="stack",
             bins=15, palette="viridis", ax=ax)
ax.set_title("Age Distribution by 'thal' Class")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
savefig(fig, "03_age_distribution.png")

# ---------------------------------------------------------------------
# 5. Max heart rate achieved (thalach) by thal class
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x="thal_label", y="thalach", order=order,
            hue="thal_label", palette="viridis", legend=False, ax=ax)
ax.set_title("Max Heart Rate Achieved (thalach) by 'thal' Class")
ax.set_xlabel("thal")
ax.set_ylabel("thalach")
savefig(fig, "04_thalach_by_thal.png")

# ---------------------------------------------------------------------
# 6. Cholesterol by thal class
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x="thal_label", y="chol", order=order,
            hue="thal_label", palette="viridis", legend=False, ax=ax)
ax.set_title("Serum Cholesterol by 'thal' Class")
ax.set_xlabel("thal")
ax.set_ylabel("chol (mg/dl)")
savefig(fig, "05_chol_by_thal.png")

# ---------------------------------------------------------------------
# 7. Heart disease target vs thal (stacked proportions)
# ---------------------------------------------------------------------
cross = pd.crosstab(df["thal_label"], df["target"])
cross = cross.reindex(order)
fig, ax = plt.subplots(figsize=(7, 5))
cross.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
ax.set_title("Heart Disease Target vs 'thal' Class")
ax.set_xlabel("thal")
ax.set_ylabel("Count")
ax.legend(title="target (1=disease present)")
plt.xticks(rotation=0)
savefig(fig, "06_target_vs_thal.png")

# ---------------------------------------------------------------------
# 8. Pairwise scatter of key numeric features, colored by thal
# ---------------------------------------------------------------------
key_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
pair_df = df[key_features + ["thal_label"]].dropna()
g = sns.pairplot(pair_df, hue="thal_label", palette="viridis",
                  diag_kind="hist", plot_kws={"alpha": 0.6, "s": 25})
g.fig.suptitle("Pairwise Relationships by 'thal' Class", y=1.02)
g.fig.savefig(os.path.join(OUT_DIR, "07_pairwise_scatter.png"),
              bbox_inches="tight", dpi=150)
plt.close(g.fig)
print(f"Saved {os.path.join(OUT_DIR, '07_pairwise_scatter.png')}")

print(f"\nAll visualizations saved in '{OUT_DIR}/'")