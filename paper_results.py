import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set style - try different style names for compatibility
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

# Set working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ============================================================================
# Configuration: Annotation Number
# ============================================================================
# Change this value to use different annotation files: 20, 30, or 40
ANNOTATION_NUMBER = 20  # Options: 20, 30, or 40

# ============================================================================
# Helper Functions
# ============================================================================

def normalize(x):
    """Normalize string: lowercase, trim, replace multiple spaces"""
    if pd.isna(x):
        return x
    return str(x).lower().strip().replace('  ', ' ').replace('\t', ' ')

def make_dmat(levels, edges):
    """
    Build shortest-path distance matrix using Floyd-Warshall algorithm
    """
    n = len(levels)
    d = np.full((n, n), np.inf)
    np.fill_diagonal(d, 0)
    
    # Set adjacency edges
    for e in edges:
        i, j = levels.index(e[0]), levels.index(e[1])
        d[i, j] = 1
        d[j, i] = 1
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i, k] + d[k, j] < d[i, j]:
                    d[i, j] = d[i, k] + d[k, j]
    
    # Replace infinite distances
    finite_max = np.max(d[np.isfinite(d)])
    d[~np.isfinite(d)] = finite_max + 1 if finite_max > 0 else 1
    
    # Create DataFrame with labels
    d_df = pd.DataFrame(d, index=levels, columns=levels)
    return d_df

def calculate_krippendorff_alpha(dat_human, dmat, move_levels, dataset_name):
    """
    Calculate Krippendorff's alpha for ordinal data using the distance matrix.
    Uses the standard formula: α = 1 - (Do / De)
    where Do is observed disagreement and De is expected disagreement.
    
    Parameters:
    - dat_human: DataFrame with columns ['Annotator', 'ID', 'Label']
    - dmat: Distance matrix DataFrame (from make_dmat)
    - move_levels: List of taxonomy levels
    - dataset_name: Name of the dataset (for error messages)
    
    Returns:
    - Krippendorff's alpha value (float) or np.nan if calculation fails
    """
    # Filter out ground truth annotators
    gt_names = ["groundtruth", "ground_truth", "gold", "gt"]
    annotators = dat_human[~dat_human["Annotator"].str.lower().isin(gt_names)]["Annotator"].unique()
    
    if len(annotators) < 2:
        print(f"Warning: Need at least 2 annotators for Krippendorff's alpha calculation in {dataset_name}")
        return np.nan
    
    # Get all items and create reliability matrix
    item_ids = sorted(dat_human["ID"].unique())
    
    # Build reliability data: list of (item_id, annotator, label) tuples
    reliability_triples = []
    for item_id in item_ids:
        item_data = dat_human[dat_human["ID"] == item_id].copy()
        item_data = item_data[~item_data["Annotator"].str.lower().isin(gt_names)]
        
        for _, row in item_data.iterrows():
            label = normalize(row["Label"])
            # Convert to string if needed (for GUG which uses numeric strings)
            label_str = str(label) if not isinstance(label, str) else label
            if label_str in move_levels:
                reliability_triples.append((item_id, row["Annotator"], label_str))
    
    if len(reliability_triples) == 0:
        print(f"Warning: No valid annotations found for Krippendorff's alpha in {dataset_name}")
        return np.nan
    
    # Calculate observed disagreement (Do)
    # Sum of squared distances between all pairs of annotations for the same item
    Do = 0.0
    n_comparable_pairs = 0
    
    # Group by item
    items_dict = {}
    for item_id, ann, label in reliability_triples:
        if item_id not in items_dict:
            items_dict[item_id] = []
        items_dict[item_id].append((ann, label))
    
    # For each item, calculate pairwise distances
    for item_id, annotations in items_dict.items():
        n_anns = len(annotations)
        if n_anns < 2:
            continue
        
        # Calculate all pairs
        for i in range(n_anns):
            for j in range(i + 1, n_anns):
                label_i = annotations[i][1]
                label_j = annotations[j][1]
                distance = dmat.loc[label_i, label_j]
                Do += distance ** 2
                n_comparable_pairs += 1
    
    if n_comparable_pairs == 0:
        print(f"Warning: No comparable annotation pairs found for Krippendorff's alpha in {dataset_name}")
        return np.nan
    
    Do = Do / n_comparable_pairs
    
    # Calculate expected disagreement (De)
    # Based on marginal distribution of labels
    label_counts = {}
    total_annotations = 0
    
    for _, _, label in reliability_triples:
        # Convert to string if needed (for GUG which uses numeric strings)
        label_str = str(label) if not isinstance(label, str) else label
        if label_str in move_levels:
            label_counts[label_str] = label_counts.get(label_str, 0) + 1
            total_annotations += 1
    
    if total_annotations == 0:
        return np.nan
    
    # Calculate expected disagreement using marginal probabilities
    De = 0.0
    label_probs = {label: count / total_annotations for label, count in label_counts.items()}
    
    # Sum over all pairs of labels (weighted by their probabilities)
    for label_u in move_levels:
        prob_u = label_probs.get(label_u, 0)
        for label_v in move_levels:
            prob_v = label_probs.get(label_v, 0)
            distance = dmat.loc[label_u, label_v]
            De += prob_u * prob_v * (distance ** 2)
    
    # Calculate alpha
    if De == 0:
        # Perfect agreement or all same label
        alpha = 1.0 if Do == 0 else np.nan
    else:
        alpha = 1 - (Do / De)
    
    return alpha

# ============================================================================
# Load and Process Human Annotation Data
# ============================================================================

print("Loading human annotation data...")

# 1. Bloom
df_bloom = pd.read_csv(f"human_annotation/Bloom-All_{ANNOTATION_NUMBER}.csv")
bloom_cols = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
df_bloom_long = df_bloom.melt(
    id_vars=[c for c in df_bloom.columns if c not in bloom_cols],
    value_vars=bloom_cols,
    var_name="Label",
    value_name="Value"
)
df_bloom_long = df_bloom_long[df_bloom_long["Value"] == 1].drop(columns=["Value"])
df_bloom_long["ID"] = pd.to_numeric(df_bloom_long["ID"])

move_levels_bloom = [l.lower() for l in bloom_cols]
adj_edges_bloom = [[move_levels_bloom[i], move_levels_bloom[i+1]] for i in range(len(move_levels_bloom)-1)]
dmat_bloom = make_dmat(move_levels_bloom, adj_edges_bloom)

dat_bloom = df_bloom_long[["Annotator", "ID", "Label"]].copy()
dat_bloom["Annotator"] = dat_bloom["Annotator"].astype(str)
dat_bloom["ID"] = dat_bloom["ID"].astype(int)
dat_bloom["Label"] = dat_bloom["Label"].apply(normalize)
dat_bloom = dat_bloom.drop_duplicates(subset=["Annotator", "ID"], keep='first')

gt_names = ["groundtruth", "ground_truth", "gold", "gt"]
gt_bloom = dat_bloom[dat_bloom["Annotator"].str.lower().isin(gt_names)][["ID", "Label"]].rename(columns={"Label": "gt"})
pred_bloom = dat_bloom[~dat_bloom["Annotator"].str.lower().isin(gt_names)].copy()
pred_bloom["Annotator"] = pred_bloom["Annotator"].replace(["A1", "A2"], "A_combined")

K_bloom = len(move_levels_bloom)
denom_bloom = K_bloom - 1 if K_bloom > 1 else 1

pred_gt_bloom = pred_bloom.merge(gt_bloom, on="ID", how="left")
pred_gt_bloom = pred_gt_bloom[
    pred_gt_bloom["gt"].notna() &
    pred_gt_bloom["Label"].isin(move_levels_bloom) &
    pred_gt_bloom["gt"].isin(move_levels_bloom)
].copy()

pred_gt_bloom["delta"] = pred_gt_bloom.apply(
    lambda row: dmat_bloom.loc[row["gt"], row["Label"]], axis=1
)
pred_gt_bloom["is_match"] = pred_gt_bloom["delta"] == 0
pred_gt_bloom["is_adj"] = pred_gt_bloom["delta"] == 1
pred_gt_bloom["is_cross"] = pred_gt_bloom["delta"] >= 2
pred_gt_bloom["sev_w"] = pred_gt_bloom["delta"] / denom_bloom

metrics_bloom_human = pred_gt_bloom.groupby("Annotator").agg({
    "delta": "count",
    "is_match": "mean",
    "is_adj": "mean",
    "is_cross": "mean",
    "sev_w": "mean"
}).rename(columns={"delta": "N", "is_match": "ACC", "is_adj": "AER", "is_cross": "CRI", "sev_w": "SWE"})
metrics_bloom_human["AS"] = 1 - metrics_bloom_human["SWE"]

# BBI calculation
pred_gt_bloom_adj = pred_gt_bloom[pred_gt_bloom["delta"] == 1].copy()
pred_gt_bloom_adj["gt_idx"] = pred_gt_bloom_adj["gt"].apply(lambda x: move_levels_bloom.index(x))
pred_gt_bloom_adj["label_idx"] = pred_gt_bloom_adj["Label"].apply(lambda x: move_levels_bloom.index(x))
pred_gt_bloom_adj["n_up"] = (pred_gt_bloom_adj["label_idx"] > pred_gt_bloom_adj["gt_idx"]).astype(int)
pred_gt_bloom_adj["n_down"] = (pred_gt_bloom_adj["label_idx"] < pred_gt_bloom_adj["gt_idx"]).astype(int)

bbi_bloom = pred_gt_bloom_adj.groupby("Annotator").agg({"n_up": "sum", "n_down": "sum"}).reset_index()
bbi_bloom["total"] = bbi_bloom["n_up"] + bbi_bloom["n_down"]
bbi_bloom["BBI"] = bbi_bloom.apply(
    lambda row: (row["n_up"] - row["n_down"]) / row["total"] if row["total"] > 0 else np.nan, axis=1
)
metrics_bloom_human = metrics_bloom_human.merge(bbi_bloom[["Annotator", "BBI"]], on="Annotator", how="left")
metrics_bloom_human["data"] = "bloom"
metrics_bloom_human = metrics_bloom_human.reset_index()

# Calculate Krippendorff's alpha for Bloom
kripp_alpha_bloom = calculate_krippendorff_alpha(dat_bloom, dmat_bloom, move_levels_bloom, "bloom")
print(f"Bloom Krippendorff's alpha: {kripp_alpha_bloom:.4f}")

# 2. MathDial
df_mathdial = pd.read_csv(f"human_annotation/MathDial-All_{ANNOTATION_NUMBER}.csv")
move_levels_mathdial = ["focus", "probing", "telling", "generic"]
adj_edges_mathdial = [[move_levels_mathdial[i], move_levels_mathdial[i+1]] for i in range(len(move_levels_mathdial)-1)]
dmat_mathdial = make_dmat(move_levels_mathdial, adj_edges_mathdial)

dat_mathdial = df_mathdial[["Annotator", "ID", "Label"]].copy()
dat_mathdial["Annotator"] = dat_mathdial["Annotator"].astype(str)
dat_mathdial["ID"] = dat_mathdial["ID"].astype(int)
dat_mathdial["Label"] = dat_mathdial["Label"].apply(normalize)
dat_mathdial = dat_mathdial.drop_duplicates(subset=["Annotator", "ID"], keep='first')

gt_mathdial = dat_mathdial[dat_mathdial["Annotator"].str.lower().isin(gt_names)][["ID", "Label"]].rename(columns={"Label": "gt"})
pred_mathdial = dat_mathdial[~dat_mathdial["Annotator"].str.lower().isin(gt_names)].copy()
pred_mathdial["Annotator"] = pred_mathdial["Annotator"].replace(["A1", "A2"], "A_combined")

K_mathdial = len(move_levels_mathdial)
denom_mathdial = K_mathdial - 1 if K_mathdial > 1 else 1

pred_gt_mathdial = pred_mathdial.merge(gt_mathdial, on="ID", how="left")
pred_gt_mathdial = pred_gt_mathdial[
    pred_gt_mathdial["gt"].notna() &
    pred_gt_mathdial["Label"].isin(move_levels_mathdial) &
    pred_gt_mathdial["gt"].isin(move_levels_mathdial)
].copy()

pred_gt_mathdial["delta"] = pred_gt_mathdial.apply(
    lambda row: dmat_mathdial.loc[row["gt"], row["Label"]], axis=1
)
pred_gt_mathdial["is_match"] = pred_gt_mathdial["delta"] == 0
pred_gt_mathdial["is_adj"] = pred_gt_mathdial["delta"] == 1
pred_gt_mathdial["is_cross"] = pred_gt_mathdial["delta"] >= 2
pred_gt_mathdial["sev_w"] = pred_gt_mathdial["delta"] / denom_mathdial

metrics_mathdial_human = pred_gt_mathdial.groupby("Annotator").agg({
    "delta": "count",
    "is_match": "mean",
    "is_adj": "mean",
    "is_cross": "mean",
    "sev_w": "mean"
}).rename(columns={"delta": "N", "is_match": "ACC", "is_adj": "AER", "is_cross": "CRI", "sev_w": "SWE"})
metrics_mathdial_human["AS"] = 1 - metrics_mathdial_human["SWE"]

pred_gt_mathdial_adj = pred_gt_mathdial[pred_gt_mathdial["delta"] == 1].copy()
pred_gt_mathdial_adj["gt_idx"] = pred_gt_mathdial_adj["gt"].apply(lambda x: move_levels_mathdial.index(x))
pred_gt_mathdial_adj["label_idx"] = pred_gt_mathdial_adj["Label"].apply(lambda x: move_levels_mathdial.index(x))
pred_gt_mathdial_adj["n_up"] = (pred_gt_mathdial_adj["label_idx"] > pred_gt_mathdial_adj["gt_idx"]).astype(int)
pred_gt_mathdial_adj["n_down"] = (pred_gt_mathdial_adj["label_idx"] < pred_gt_mathdial_adj["gt_idx"]).astype(int)

bbi_mathdial = pred_gt_mathdial_adj.groupby("Annotator").agg({"n_up": "sum", "n_down": "sum"}).reset_index()
bbi_mathdial["total"] = bbi_mathdial["n_up"] + bbi_mathdial["n_down"]
bbi_mathdial["BBI"] = bbi_mathdial.apply(
    lambda row: (row["n_up"] - row["n_down"]) / row["total"] if row["total"] > 0 else np.nan, axis=1
)
metrics_mathdial_human = metrics_mathdial_human.merge(bbi_mathdial[["Annotator", "BBI"]], on="Annotator", how="left")
metrics_mathdial_human["data"] = "mathdial"
metrics_mathdial_human = metrics_mathdial_human.reset_index()

# Calculate Krippendorff's alpha for MathDial
kripp_alpha_mathdial = calculate_krippendorff_alpha(dat_mathdial, dmat_mathdial, move_levels_mathdial, "mathdial")
print(f"MathDial Krippendorff's alpha: {kripp_alpha_mathdial:.4f}")

# 3. Uptake
df_uptake = pd.read_csv(f"human_annotation/Uptake-All_{ANNOTATION_NUMBER}.csv")
move_levels_uptake = ["low", "mid", "high"]
adj_edges_uptake = [[move_levels_uptake[i], move_levels_uptake[i+1]] for i in range(len(move_levels_uptake)-1)]
dmat_uptake = make_dmat(move_levels_uptake, adj_edges_uptake)

dat_uptake = df_uptake[["Annotator", "ID", "uptake_majority"]].rename(columns={"uptake_majority": "Label"}).copy()
dat_uptake["Annotator"] = dat_uptake["Annotator"].astype(str)
dat_uptake["ID"] = dat_uptake["ID"].astype(int)
dat_uptake["Label"] = dat_uptake["Label"].apply(normalize)
dat_uptake = dat_uptake.drop_duplicates(subset=["Annotator", "ID"], keep='first')

gt_uptake = dat_uptake[dat_uptake["Annotator"].str.lower().isin(gt_names)][["ID", "Label"]].rename(columns={"Label": "gt"})
pred_uptake = dat_uptake[~dat_uptake["Annotator"].str.lower().isin(gt_names)].copy()
pred_uptake["Annotator"] = pred_uptake["Annotator"].replace(["A1", "A2"], "A_combined")

K_uptake = len(move_levels_uptake)
denom_uptake = K_uptake - 1 if K_uptake > 1 else 1

pred_gt_uptake = pred_uptake.merge(gt_uptake, on="ID", how="left")
pred_gt_uptake = pred_gt_uptake[
    pred_gt_uptake["gt"].notna() &
    pred_gt_uptake["Label"].isin(move_levels_uptake) &
    pred_gt_uptake["gt"].isin(move_levels_uptake)
].copy()

pred_gt_uptake["delta"] = pred_gt_uptake.apply(
    lambda row: dmat_uptake.loc[row["gt"], row["Label"]], axis=1
)
pred_gt_uptake["is_match"] = pred_gt_uptake["delta"] == 0
pred_gt_uptake["is_adj"] = pred_gt_uptake["delta"] == 1
pred_gt_uptake["is_cross"] = pred_gt_uptake["delta"] >= 2
pred_gt_uptake["sev_w"] = pred_gt_uptake["delta"] / denom_uptake

metrics_uptake_human = pred_gt_uptake.groupby("Annotator").agg({
    "delta": "count",
    "is_match": "mean",
    "is_adj": "mean",
    "is_cross": "mean",
    "sev_w": "mean"
}).rename(columns={"delta": "N", "is_match": "ACC", "is_adj": "AER", "is_cross": "CRI", "sev_w": "SWE"})
metrics_uptake_human["AS"] = 1 - metrics_uptake_human["SWE"]

pred_gt_uptake_adj = pred_gt_uptake[pred_gt_uptake["delta"] == 1].copy()
pred_gt_uptake_adj["gt_idx"] = pred_gt_uptake_adj["gt"].apply(lambda x: move_levels_uptake.index(x))
pred_gt_uptake_adj["label_idx"] = pred_gt_uptake_adj["Label"].apply(lambda x: move_levels_uptake.index(x))
pred_gt_uptake_adj["n_up"] = (pred_gt_uptake_adj["label_idx"] > pred_gt_uptake_adj["gt_idx"]).astype(int)
pred_gt_uptake_adj["n_down"] = (pred_gt_uptake_adj["label_idx"] < pred_gt_uptake_adj["gt_idx"]).astype(int)

bbi_uptake = pred_gt_uptake_adj.groupby("Annotator").agg({"n_up": "sum", "n_down": "sum"}).reset_index()
bbi_uptake["total"] = bbi_uptake["n_up"] + bbi_uptake["n_down"]
bbi_uptake["BBI"] = bbi_uptake.apply(
    lambda row: (row["n_up"] - row["n_down"]) / row["total"] if row["total"] > 0 else np.nan, axis=1
)
metrics_uptake_human = metrics_uptake_human.merge(bbi_uptake[["Annotator", "BBI"]], on="Annotator", how="left")
metrics_uptake_human["data"] = "uptake"
metrics_uptake_human = metrics_uptake_human.reset_index()

# Calculate Krippendorff's alpha for Uptake
kripp_alpha_uptake = calculate_krippendorff_alpha(dat_uptake, dmat_uptake, move_levels_uptake, "uptake")
print(f"Uptake Krippendorff's alpha: {kripp_alpha_uptake:.4f}")

# 4. GUG
df_gug = pd.read_csv("human_annotation/GUG-All_20.csv")
dat_gug = df_gug[["Annotator", "ID", "Label"]].copy()
dat_gug["Annotator"] = dat_gug["Annotator"].astype(str)
dat_gug["ID"] = dat_gug["ID"].astype(int)
# Convert Label to string for consistency with other datasets
dat_gug["Label"] = dat_gug["Label"].astype(str)
dat_gug = dat_gug.drop_duplicates(subset=["Annotator", "ID"], keep='first')

# Create distance matrix for GUG (needed for Krippendorff's alpha calculation)
move_levels_gug = ["1", "2", "3", "4"]
adj_edges_gug = [[move_levels_gug[i], move_levels_gug[i+1]] for i in range(len(move_levels_gug)-1)]
dmat_gug = make_dmat(move_levels_gug, adj_edges_gug)

gt_gug = dat_gug[dat_gug["Annotator"].str.lower().isin(gt_names)][["ID", "Label"]].rename(columns={"Label": "gt"})
pred_gug = dat_gug[~dat_gug["Annotator"].str.lower().isin(gt_names)].copy()
pred_gug["Annotator"] = pred_gug["Annotator"].replace(["A1", "A2"], "A_combined")

pred_gt_gug = pred_gug.merge(gt_gug, on="ID", how="left")
pred_gt_gug = pred_gt_gug[pred_gt_gug["gt"].notna()].copy()
# Convert to numeric for signed calculation (GUG uses string labels "1", "2", "3", "4")
pred_gt_gug["signed"] = pred_gt_gug["Label"].astype(int) - pred_gt_gug["gt"].astype(int)
pred_gt_gug["delta"] = pred_gt_gug["signed"].abs()

K_gug = len(sorted(pred_gt_gug["gt"].unique()))
denom_gug = K_gug - 1 if K_gug > 1 else 1
pred_gt_gug["w_delta"] = pred_gt_gug["delta"] / denom_gug

metrics_gug_human = pred_gt_gug.groupby("Annotator").agg({
    "delta": "count",
    "Label": lambda x: (x == pred_gt_gug.loc[x.index, "gt"]).mean(),
    "delta": lambda x: (x == 1).mean(),
    "delta": lambda x: (x >= 2).mean(),
    "w_delta": "mean"
}).rename(columns={"delta": "N"})

# Fix aggregation for GUG
metrics_gug_human = pred_gt_gug.groupby("Annotator").agg({
    "Label": "count",
    "w_delta": "mean"
}).rename(columns={"Label": "N", "w_delta": "SWE"})
metrics_gug_human["ACC"] = pred_gt_gug.groupby("Annotator").apply(lambda x: (x["Label"] == x["gt"]).mean()).values
metrics_gug_human["AER"] = pred_gt_gug.groupby("Annotator").apply(lambda x: (x["delta"] == 1).mean()).values
metrics_gug_human["CRI"] = pred_gt_gug.groupby("Annotator").apply(lambda x: (x["delta"] >= 2).mean()).values
metrics_gug_human["AS"] = 1 - metrics_gug_human["SWE"]

pred_gt_gug_adj = pred_gt_gug[pred_gt_gug["delta"] == 1].copy()
pred_gt_gug_adj["n_up"] = (pred_gt_gug_adj["signed"] > 0).astype(int)
pred_gt_gug_adj["n_down"] = (pred_gt_gug_adj["signed"] < 0).astype(int)

bbi_gug = pred_gt_gug_adj.groupby("Annotator").agg({"n_up": "sum", "n_down": "sum"}).reset_index()
bbi_gug["total"] = bbi_gug["n_up"] + bbi_gug["n_down"]
bbi_gug["BBI"] = bbi_gug.apply(
    lambda row: (row["n_up"] - row["n_down"]) / row["total"] if row["total"] > 0 else np.nan, axis=1
)
metrics_gug_human = metrics_gug_human.merge(bbi_gug[["Annotator", "BBI"]], on="Annotator", how="left")
metrics_gug_human["data"] = "gug"
metrics_gug_human = metrics_gug_human.reset_index()

# Calculate Krippendorff's alpha for GUG
kripp_alpha_gug = calculate_krippendorff_alpha(dat_gug, dmat_gug, move_levels_gug, "gug")
print(f"GUG Krippendorff's alpha: {kripp_alpha_gug:.4f}")

# Combine human metrics
metrics_human = pd.concat([
    metrics_bloom_human,
    metrics_mathdial_human,
    metrics_uptake_human,
    metrics_gug_human
], ignore_index=True)
metrics_human["source"] = "Human"

print("Human annotation data processed.")

# ============================================================================
# Load and Process Model Performance Data
# ============================================================================

print("Loading model performance data...")

# Load model full data and compute metrics
bloom_full = pd.read_csv("model_annotation/bloom_full.csv")
mathdial_full = pd.read_csv("model_annotation/mathdial_full.csv")
uptake_full = pd.read_csv("model_annotation/uptake_full.csv")
gug_full = pd.read_csv("model_annotation/gug_full.csv")

# Process Bloom model metrics with graph distance
bloom_full["human_category"] = bloom_full["human_category"].astype(str).apply(normalize)
bloom_full["model_category"] = bloom_full["model_category"].astype(str).apply(normalize)
bloom_full["human_category"] = bloom_full["human_category"].apply(lambda x: x if x in move_levels_bloom else np.nan)
bloom_full["model_category"] = bloom_full["model_category"].apply(lambda x: x if x in move_levels_bloom else np.nan)
# Clean Technique names
bloom_full["Technique"] = bloom_full["Technique"].astype(str).str.replace(r'DSPy_.*?_(\d+)', r'DSPy_\1', regex=True)

bloom_model = bloom_full[bloom_full["human_category"].notna() & bloom_full["model_category"].notna()].copy()
bloom_model["delta"] = bloom_model.apply(
    lambda row: dmat_bloom.loc[row["human_category"], row["model_category"]], axis=1
)
bloom_model["is_match"] = bloom_model["delta"] == 0
bloom_model["is_adj"] = bloom_model["delta"] == 1
bloom_model["is_cross"] = bloom_model["delta"] >= 2
bloom_model["sev_w"] = bloom_model["delta"] / denom_bloom

bloom_metrics = bloom_model.groupby(["model", "Technique"]).agg({
    "delta": "count",
    "is_match": "mean",
    "is_adj": "mean",
    "is_cross": "mean",
    "sev_w": "mean"
}).rename(columns={"delta": "N", "is_match": "ACC", "is_adj": "AER", "is_cross": "CRI", "sev_w": "SWE"})
bloom_metrics["AS"] = 1 - bloom_metrics["SWE"]
bloom_metrics["data"] = "bloom"
bloom_metrics = bloom_metrics.reset_index()

# Process MathDial model metrics
mathdial_full["ground_truth_move"] = mathdial_full["ground_truth_move"].apply(normalize)
mathdial_full["predicted_move"] = mathdial_full["predicted_move"].apply(normalize)

mathdial_model = mathdial_full[
    mathdial_full["ground_truth_move"].isin(move_levels_mathdial) &
    mathdial_full["predicted_move"].isin(move_levels_mathdial)
].copy()
mathdial_model["delta"] = mathdial_model.apply(
    lambda row: dmat_mathdial.loc[row["ground_truth_move"], row["predicted_move"]], axis=1
)
mathdial_model["is_match"] = mathdial_model["delta"] == 0
mathdial_model["is_adj"] = mathdial_model["delta"] == 1
mathdial_model["is_cross"] = mathdial_model["delta"] >= 2
mathdial_model["sev_w"] = mathdial_model["delta"] / denom_mathdial

mathdial_metrics = mathdial_model.groupby(["model", "Technique"]).agg({
    "delta": "count",
    "is_match": "mean",
    "is_adj": "mean",
    "is_cross": "mean",
    "sev_w": "mean"
}).rename(columns={"delta": "N", "is_match": "ACC", "is_adj": "AER", "is_cross": "CRI", "sev_w": "SWE"})
mathdial_metrics["AS"] = 1 - mathdial_metrics["SWE"]
mathdial_metrics["data"] = "mathdial"
mathdial_metrics = mathdial_metrics.reset_index()

# Process Uptake model metrics
uptake_full["ground_truth"] = uptake_full["ground_truth"].apply(normalize)
uptake_full["prediction"] = uptake_full["prediction"].apply(normalize)

uptake_model = uptake_full[
    uptake_full["ground_truth"].isin(move_levels_uptake) &
    uptake_full["prediction"].isin(move_levels_uptake)
].copy()
uptake_model["delta"] = uptake_model.apply(
    lambda row: dmat_uptake.loc[row["ground_truth"], row["prediction"]], axis=1
)
uptake_model["is_match"] = uptake_model["delta"] == 0
uptake_model["is_adj"] = uptake_model["delta"] == 1
uptake_model["is_cross"] = uptake_model["delta"] >= 2
uptake_model["sev_w"] = uptake_model["delta"] / denom_uptake

uptake_metrics = uptake_model.groupby(["model", "Technique"]).agg({
    "delta": "count",
    "is_match": "mean",
    "is_adj": "mean",
    "is_cross": "mean",
    "sev_w": "mean"
}).rename(columns={"delta": "N", "is_match": "ACC", "is_adj": "AER", "is_cross": "CRI", "sev_w": "SWE"})
uptake_metrics["AS"] = 1 - uptake_metrics["SWE"]
uptake_metrics["data"] = "uptake"
uptake_metrics = uptake_metrics.reset_index()

# Process GUG model metrics
# Note: dmat_gug was already created above for human metrics calculation

gug_full["ground_truth"] = gug_full["ground_truth"].apply(normalize)
gug_full["prediction"] = gug_full["prediction"].apply(normalize)

gug_model = gug_full[
    gug_full["ground_truth"].isin(move_levels_gug) &
    gug_full["prediction"].isin(move_levels_gug)
].copy()
gug_model["delta"] = gug_model.apply(
    lambda row: dmat_gug.loc[row["ground_truth"], row["prediction"]], axis=1
)
gug_model["is_match"] = gug_model["delta"] == 0
gug_model["is_adj"] = gug_model["delta"] == 1
gug_model["is_cross"] = gug_model["delta"] >= 2

K_gug_model = len(move_levels_gug)
denom_gug_model = K_gug_model - 1 if K_gug_model > 1 else 1
gug_model["sev_w"] = gug_model["delta"] / denom_gug_model

gug_metrics = gug_model.groupby(["model", "Technique"]).agg({
    "delta": "count",
    "is_match": "mean",
    "is_adj": "mean",
    "is_cross": "mean",
    "sev_w": "mean"
}).rename(columns={"delta": "N", "is_match": "ACC", "is_adj": "AER", "is_cross": "CRI", "sev_w": "SWE"})
gug_metrics["AS"] = 1 - gug_metrics["SWE"]
gug_metrics["data"] = "gug"
gug_metrics = gug_metrics.reset_index()

# Combine all model metrics
all_metrics = pd.concat([bloom_metrics, mathdial_metrics, uptake_metrics, gug_metrics], ignore_index=True)
# Ensure clean index
all_metrics = all_metrics.reset_index(drop=True)

print("Model performance data processed.")

# ============================================================================
# Data Preparation for Analysis
# ============================================================================

print("Preparing data for analysis...")

# Add human metrics
# Ensure Annotator column is renamed to model
if "Annotator" in metrics_human.columns:
    metrics_human = metrics_human.rename(columns={"Annotator": "model"})

# Reset index if model is in the index
if metrics_human.index.name == "Annotator":
    metrics_human = metrics_human.reset_index()
    if "Annotator" in metrics_human.columns:
        metrics_human = metrics_human.rename(columns={"Annotator": "model"})

# Add Technique and model columns
metrics_human["Technique"] = "Human"
metrics_human["model"] = "Human"

# Ensure we have all required columns matching all_metrics structure
# all_metrics has: model, Technique, N, ACC, AER, CRI, SWE, AS, data
required_cols = ["model", "Technique", "N", "ACC", "AER", "CRI", "SWE", "AS", "data"]
# Make sure all columns exist, fill missing ones
for col in required_cols:
    if col not in metrics_human.columns:
        if col == "N":
            metrics_human[col] = 0  # Will be recalculated if needed
        else:
            metrics_human[col] = 0.0

# Select only the required columns in the right order
metrics_human = metrics_human[required_cols]
# Ensure clean index and no duplicates
metrics_human = metrics_human.reset_index(drop=True)
metrics_human = metrics_human.drop_duplicates()

# Ensure all_metrics also has clean index
all_metrics = all_metrics.reset_index(drop=True)

# Combine - ensure both have same column order and types
all_metrics_cols = list(all_metrics.columns)
# Make sure metrics_human has all the same columns
for col in all_metrics_cols:
    if col not in metrics_human.columns:
        metrics_human[col] = 0.0 if all_metrics[col].dtype in [np.float64, np.float32, float] else 0

# Reorder metrics_human to match all_metrics column order
metrics_human = metrics_human[all_metrics_cols]

# Ensure column types match
for col in all_metrics_cols:
    if all_metrics[col].dtype != metrics_human[col].dtype:
        try:
            metrics_human[col] = metrics_human[col].astype(all_metrics[col].dtype)
        except:
            pass  # If conversion fails, keep original

all = pd.concat([all_metrics, metrics_human], ignore_index=True, sort=False)

# Calculate Krippendorff's alpha from -All_ files (no need for separate agreement files)
# Create agreement dataframe from calculated alpha values
agreement = pd.DataFrame({
    "data": ["bloom", "mathdial", "uptake", "gug"],
    "kripp_alpha": [kripp_alpha_bloom, kripp_alpha_mathdial, kripp_alpha_uptake, kripp_alpha_gug]
})

all = all.merge(agreement, on="data", how="left")
print("Krippendorff's alpha calculated from -All_ files and merged into dataset.")

# Calculate percentages
all["perc_AER"] = all["AER"] / (1 - all["ACC"] + 1e-10)
all["perc_CRI"] = all["CRI"] / (1 - all["ACC"] + 1e-10)
all = all.drop(columns=["N"])

# Create human baseline
human_baseline = all[all["model"] == "Human"].copy()
human_baseline = human_baseline.drop(columns=["model", "Technique"])
human_baseline.columns = ["H_" + col if col != "data" else col for col in human_baseline.columns]

# Join with model data
df_all = all[all["model"] != "Human"].copy()
df_all = df_all.merge(human_baseline, on="data", how="left")

# Rename model columns (add M_ prefix to metric columns that don't already have it)
model_cols_to_rename = {
    "ACC": "M_ACC",
    "AER": "M_AER", 
    "CRI": "M_CRI",
    "SWE": "M_SWE",
    "AS": "M_AS",
    "perc_AER": "M_perc_AER",
    "perc_CRI": "M_perc_CRI"
}

# Only rename if the column exists and doesn't already have M_ prefix
for old_col, new_col in model_cols_to_rename.items():
    if old_col in df_all.columns and new_col not in df_all.columns:
        df_all = df_all.rename(columns={old_col: new_col})

# Now select and reorder columns to match expected structure
# Get all current columns
current_cols = list(df_all.columns)

# Build the expected column list based on what we have
expected_cols = []
if "model" in current_cols:
    expected_cols.append("model")
if "Technique" in current_cols:
    expected_cols.append("Technique")
if "M_ACC" in current_cols:
    expected_cols.append("M_ACC")
if "M_AER" in current_cols:
    expected_cols.append("M_AER")
if "M_CRI" in current_cols:
    expected_cols.append("M_CRI")
if "M_SWE" in current_cols:
    expected_cols.append("M_SWE")
if "M_AS" in current_cols:
    expected_cols.append("M_AS")
if "data" in current_cols:
    expected_cols.append("data")
if "M_perc_AER" in current_cols:
    expected_cols.append("M_perc_AER")
if "M_perc_CRI" in current_cols:
    expected_cols.append("M_perc_CRI")

# Add all H_ columns
h_cols = [c for c in current_cols if c.startswith("H_")]
expected_cols.extend(sorted(h_cols))

# Select only the columns we want, in the order we want
df_all = df_all[expected_cols]

# Calculate error decomposition
df_all["AER_task"] = (df_all["H_AER"] / (df_all["M_AER"] + df_all["H_AER"] + 1e-10)) * df_all["M_AER"]
df_all["CRI_task"] = (df_all["H_CRI"] / (df_all["M_CRI"] + df_all["H_CRI"] + 1e-10)) * df_all["M_CRI"]
df_all["AER_model"] = df_all["M_AER"] - df_all["AER_task"]
df_all["CRI_model"] = df_all["M_CRI"] - df_all["CRI_task"]

print("Data preparation complete.")

# ============================================================================
# Visualization 1: RQ1 Human Error
# ============================================================================

print("Creating Fig2_Distribution_of_error_types_observed_in_human_annotations.png...")

# Select human columns (H_kripp_alpha may not exist if agreement files are missing)
human_cols = ["data", "H_ACC", "H_AER", "H_CRI", "H_SWE", "H_AS", "H_perc_AER", "H_perc_CRI"]
if "H_kripp_alpha" in df_all.columns:
    human_cols.append("H_kripp_alpha")
human_all = df_all[human_cols].drop_duplicates()

df_long = human_all[["data", "H_ACC", "H_AER", "H_CRI"]].melt(
    id_vars=["data"],
    value_vars=["H_ACC", "H_AER", "H_CRI"],
    var_name="Metric",
    value_name="Value"
)

df_long["Metric"] = df_long["Metric"].map({
    "H_AER": "Boundary Ambiguity Error",
    "H_CRI": "Conceptual Misidentification Error",
    "H_ACC": "Correct Annotation"
})

df_long["data"] = df_long["data"].map({
    "mathdial": "MathDial",
    "gug": "GuG",
    "bloom": "Bloom",
    "uptake": "Uptake"
})

# Set data order for x-axis: mathdial, gug, bloom, uptake
data_order = ["MathDial", "GuG", "Bloom", "Uptake"]
df_long["data"] = pd.Categorical(df_long["data"], categories=data_order, ordered=True)

# Reorder for stacking - bottom to top: Correct, Conceptual, Boundary
# In matplotlib, first column in list goes at bottom of stack
metric_order = ["Correct Annotation", "Conceptual Misidentification Error", "Boundary Ambiguity Error"]
df_long["Metric"] = pd.Categorical(df_long["Metric"], 
                                   categories=metric_order,
                                   ordered=True)

fig, ax = plt.subplots(figsize=(4, 2))
df_long_pivot = df_long.pivot_table(index="data", columns="Metric", values="Value", aggfunc="first")
# Reorder index (x-axis) to match desired order
df_long_pivot = df_long_pivot.reindex(data_order)
# Reorder columns for stacking (bottom to top: Correct at bottom, Boundary at top)
df_long_pivot = df_long_pivot[metric_order]
# Colors in order: green (Correct), yellow (Conceptual), orange (Boundary)
df_long_pivot.plot(kind="bar", stacked=True, ax=ax, 
                   color=["forestgreen", "#FCD531", "#F38C23"],
                   width=0.8)

ax.set_xlabel("", fontsize=7, fontweight="bold")
ax.set_ylabel("", fontsize=7, fontweight="bold")
ax.set_title("", fontsize=8, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.tick_params(labelsize=6.5)
ax.tick_params(axis='x', labelsize=6.5, rotation=0)
ax.legend(title="Error Types (Human Annotators)", fontsize=6.5, title_fontsize=7, 
          frameon=True, loc='best')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("Fig2_Distribution_of_error_types_observed_in_human_annotations.png", dpi=1000, bbox_inches='tight')
plt.close()

print("Fig2_Distribution_of_error_types_observed_in_human_annotations.png saved.")

# ============================================================================
# Visualization 2: RQ1 Model Error Decomposition
# ============================================================================

print("Creating Fig_3_Comparison_of_Error_Decomposition_Between_Two_Models.png...")

zero_long = df_all[df_all["Technique"] == "Zero-shot"].copy()
zero_long = zero_long[["model", "data", "M_ACC", "AER_task", "CRI_task", "AER_model", "CRI_model"]].melt(
    id_vars=["model", "data"],
    value_vars=["M_ACC", "AER_task", "CRI_task", "AER_model", "CRI_model"],
    var_name="Component",
    value_name="Value"
)

zero_long["total"] = zero_long.groupby(["model", "data"])["Value"].transform("sum")
zero_long["prop"] = zero_long["Value"] / zero_long["total"]
zero_long["pct_label"] = zero_long["prop"].apply(lambda x: f"{x:.1%}" if x >= 0.04 else "")

zero_long["data"] = zero_long["data"].map({
    "mathdial": "MathDial",
    "gug": "GuG",
    "bloom": "Bloom",
    "uptake": "Uptake"
})
# Set data order for x-axis: mathdial, gug, bloom, uptake
data_order_model = ["MathDial", "GuG", "Bloom", "Uptake"]
zero_long["data"] = pd.Categorical(zero_long["data"], categories=data_order_model, ordered=True)

zero_long["Component"] = zero_long["Component"].map({
    "AER_task": "Boundary Ambiguity Error (Task)",
    "CRI_task": "Conceptual Misidentification Error (Task)",
    "AER_model": "Boundary Ambiguity Error (Model)",
    "CRI_model": "Conceptual Misidentification Error (Model)",
    "M_ACC": "Correct Annotation"
})
# Set component order for stacking (bottom to top): M_ACC, CRI_model, AER_model, CRI_task, AER_task (REVERSED)
component_order_model = ["Correct Annotation",
                        "Conceptual Misidentification Error (Model)",
                        "Boundary Ambiguity Error (Model)",
                        "Conceptual Misidentification Error (Task)",
                        "Boundary Ambiguity Error (Task)"]
zero_long["Component"] = pd.Categorical(zero_long["Component"], categories=component_order_model, ordered=True)

comp_colors = {
    "Correct Annotation": "forestgreen",
    "Boundary Ambiguity Error (Task)": "steelblue",
    "Conceptual Misidentification Error (Task)": "skyblue",
    "Boundary Ambiguity Error (Model)": "firebrick",
    "Conceptual Misidentification Error (Model)": "indianred"
}

fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.8), sharey=True)
# Ensure we iterate in the correct data order
for idx, data_name in enumerate(data_order_model):
    data_group = zero_long[zero_long["data"] == data_name]
    if len(data_group) == 0:
        continue
    ax = axes[idx]
    data_pivot = data_group.pivot_table(
        index="model", columns="Component", values="prop", aggfunc="first"
    )
    # Use the predefined component order
    data_pivot = data_pivot[[c for c in component_order_model if c in data_pivot.columns]]
    data_pivot.plot(kind="bar", stacked=True, ax=ax, color=[comp_colors[c] for c in data_pivot.columns],
                   width=0.8)
    
    # Add percentage labels (simplified - show labels for segments >= 4%)
    try:
        for container in ax.containers:
            labels = []
            for i, bar in enumerate(container):
                height = bar.get_height()
                if height >= 0.04:
                    labels.append(f"{height:.1%}")
                else:
                    labels.append("")
            ax.bar_label(container, labels=labels, label_type='center', color='white', fontsize=6, fontweight='bold')
    except:
        pass  # Skip labels if there's an issue
    
    ax.set_title(data_name, fontsize=7, fontweight="bold")
    ax.set_xlabel("", fontsize=7, fontweight="bold")
    ax.set_ylabel("", fontsize=7, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.tick_params(labelsize=6.5)
    ax.tick_params(axis='x', labelsize=6.5, rotation=0)
    ax.legend().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add legend to the right
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Error Decomposition", loc='center right', 
          fontsize=6.5, title_fontsize=7, frameon=True)

plt.tight_layout()
plt.savefig("Fig_3_Comparison_of_Error_Decomposition_Between_Two_Models.png", dpi=1000, bbox_inches='tight')
plt.close()

print("Fig_3_Comparison_of_Error_Decomposition_Between_Two_Models.png saved.")

# ============================================================================
# Visualization 3: RQ1 Prompting Techniques
# ============================================================================

print("Creating Fig_4_Error_Decomposition_Across_Prompting_Strategies.png...")

two_long = df_all[~df_all["Technique"].str.contains("APO", case=False, na=False)].copy()
two_long = two_long[["model", "data", "Technique", "M_ACC", "AER_task", "CRI_task", "AER_model", "CRI_model"]].melt(
    id_vars=["model", "data", "Technique"],
    value_vars=["M_ACC", "AER_task", "CRI_task", "AER_model", "CRI_model"],
    var_name="Component",
    value_name="Value"
)

two_long["total"] = two_long.groupby(["model", "data", "Technique"])["Value"].transform("sum")
two_long["prop"] = two_long["Value"] / two_long["total"]

two_long["data"] = two_long["data"].map({
    "mathdial": "MathDial",
    "gug": "GuG",
    "bloom": "Bloom",
    "uptake": "Uptake"
})
# Set data order for x-axis: mathdial, gug, bloom, uptake
data_order_prompting = ["MathDial", "GuG", "Bloom", "Uptake"]
two_long["data"] = pd.Categorical(two_long["data"], categories=data_order_prompting, ordered=True)

technique_order = ["Zero-shot", "Few-shot", "Auto-CoT", "Self-Consistency", 
                   "Active Prompting", "DSPy_100", "DSPy_200", "DSPy_400"]
two_long["Technique"] = pd.Categorical(two_long["Technique"], categories=technique_order, ordered=True)

two_long["Component"] = two_long["Component"].map({
    "AER_task": "Boundary Ambiguity Error (Task)",
    "CRI_task": "Conceptual Misidentification Error (Task)",
    "AER_model": "Boundary Ambiguity Error (Model)",
    "CRI_model": "Conceptual Misidentification Error (Model)",
    "M_ACC": "Correct Annotation"
})
# Set component order for stacking (same as model plot)
two_long["Component"] = pd.Categorical(two_long["Component"], categories=component_order_model, ordered=True)

# Create facet grid - use sorted models and ordered data
models = sorted(two_long["model"].unique())
datas = data_order_prompting  # Use the ordered data list
fig, axes = plt.subplots(len(models), len(datas), figsize=(8, 3.5), sharey=True)

# Handle case where axes might be 1D
if len(models) == 1:
    axes = axes.reshape(1, -1)
elif len(datas) == 1:
    axes = axes.reshape(-1, 1)
else:
    axes = axes.reshape(len(models), len(datas))

for i, model in enumerate(models):
    for j, data in enumerate(datas):
        if len(models) == 1:
            ax = axes[0, j]
        elif len(datas) == 1:
            ax = axes[i, 0]
        else:
            ax = axes[i, j]
        subset = two_long[(two_long["model"] == model) & (two_long["data"] == data)]
        if len(subset) > 0:
            subset_pivot = subset.pivot_table(
                index="Technique", columns="Component", values="prop", aggfunc="first"
            )
            # Use the predefined component order
            available_cols = [c for c in component_order_model if c in subset_pivot.columns]
            subset_pivot = subset_pivot[available_cols]
            # Ensure Technique order is maintained
            subset_pivot = subset_pivot.reindex(technique_order)
            subset_pivot.plot(kind="bar", stacked=True, ax=ax, 
                            color=[comp_colors[c] for c in subset_pivot.columns],
                            width=0.8)
        
        if i == 0:
            ax.set_title(data, fontsize=7, fontweight="bold")
        if j == 0:
            ax.set_ylabel(model, fontsize=7, fontweight="bold", rotation=90)
        ax.set_xlabel("", fontsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.tick_params(labelsize=6.5)
        ax.tick_params(axis='x', labelsize=6.5, rotation=45)
        # Set horizontal alignment for x-axis labels
        for label in ax.get_xticklabels():
            label.set_ha('right')
        ax.legend().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False, axis='x')

# Add legend
if len(models) == 1:
    handles, labels = axes[0, 0].get_legend_handles_labels()
elif len(datas) == 1:
    handles, labels = axes[0, 0].get_legend_handles_labels()
else:
    handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title="Error Decomposition", loc='center right',
          fontsize=6.5, title_fontsize=7, frameon=True)

plt.tight_layout()
plt.savefig("Fig_4_Error_Decomposition_Across_Prompting_Strategies.png", dpi=1000, bbox_inches='tight')
plt.close()

print("Fig_4_Error_Decomposition_Across_Prompting_Strategies.png saved.")

# ============================================================================
# Visualization 4: RQ2 Baseline
# ============================================================================

print("Creating Fig_5_Baseline_Zero_Shot_Model_Accuracy_vs_Task_Inherent_Errors.png...")

df = df_all[df_all["Technique"] == "Zero-shot"].copy()

# Check if H_kripp_alpha has any non-NaN values
has_kripp_alpha = "H_kripp_alpha" in df.columns and df["H_kripp_alpha"].notna().any()

# Select variables for plotting (exclude H_kripp_alpha if all NaN)
value_vars = ["H_ACC", "H_AER", "H_CRI"]
if has_kripp_alpha:
    value_vars.insert(1, "H_kripp_alpha")  # Insert after H_ACC

plot_df = df[["data", "model", "H_ACC", "H_AER", "H_CRI", "H_kripp_alpha", "M_ACC"]].melt(
    id_vars=["data", "model", "M_ACC"],
    value_vars=value_vars,
    var_name="Variable",
    value_name="HumanValue"
)

plot_df["Variable"] = plot_df["Variable"].map({
    "H_ACC": "H-Accuracy",
    "H_kripp_alpha": "H-Agree",
    "H_AER": "H-Boundary Ambiguity",
    "H_CRI": "H-Conceptual Misidentification"
})

# Build variable order based on what we have
variable_order_baseline = ["H-Accuracy"]
if has_kripp_alpha:
    variable_order_baseline.append("H-Agree")
variable_order_baseline.extend(["H-Boundary Ambiguity", "H-Conceptual Misidentification"])
plot_df["Variable"] = pd.Categorical(plot_df["Variable"], categories=variable_order_baseline, ordered=True)

# Adjust number of subplots based on available variables
n_plots = len(variable_order_baseline)
fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4), sharey=True)
if n_plots == 1:
    axes = [axes]  # Make it iterable

# Ensure we iterate in the correct variable order
for idx, var_name in enumerate(variable_order_baseline):
    var_group = plot_df[plot_df["Variable"] == var_name]
    if len(var_group) == 0:
        continue
    ax = axes[idx]
    
    # Get unique models and datasets for colors and shapes
    models = sorted(var_group["model"].unique())
    # Use consistent dataset order to match R's factor levels
    # R orders factors alphabetically or by appearance, so: bloom, gug, mathdial, uptake
    dataset_order = ["bloom", "gug", "mathdial", "uptake"]
    datasets_ordered = [d for d in dataset_order if d in var_group["data"].unique()]
    datasets_ordered.extend([d for d in var_group["data"].unique() if d not in datasets_ordered])
    
    # Use ggplot2's default discrete color palette (Set1 from RColorBrewer)
    # Colors: Red, Blue, Green, Purple
    ggplot_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF']
    color_map = {ds: ggplot_colors[i % len(ggplot_colors)] for i, ds in enumerate(datasets_ordered)}
    # Shape map for models (circles and triangles)
    shape_map = {mod: ['o', '^', 's', 'D'][i % 4] for i, mod in enumerate(models)}
    
    # Plot points by dataset (color) and model (shape)
    for data_name in datasets_ordered:
        for model_name in models:
            subset = var_group[(var_group["data"] == data_name) & (var_group["model"] == model_name)].copy()
            # Remove NaN and infinite values
            subset = subset[
                subset["HumanValue"].notna() & 
                subset["M_ACC"].notna() &
                np.isfinite(subset["HumanValue"]) &
                np.isfinite(subset["M_ACC"])
            ]
            
            if len(subset) > 0:
                # Use color for dataset, shape for model
                label = data_name if model_name == models[0] else ""  # Only label first model per dataset
                ax.scatter(subset["HumanValue"], subset["M_ACC"], 
                          c=[color_map[data_name]], marker=shape_map[model_name],
                          label=label, s=100, alpha=0.7, edgecolors='none')
    
    # Fit regression lines by MODEL (not by dataset)
    for model_name in models:
        model_subset = var_group[var_group["model"] == model_name].copy()
        model_subset = model_subset[
            model_subset["HumanValue"].notna() & 
            model_subset["M_ACC"].notna() &
            np.isfinite(model_subset["HumanValue"]) &
            np.isfinite(model_subset["M_ACC"])
        ]
        
        if len(model_subset) > 1:
            try:
                z = np.polyfit(model_subset["HumanValue"], model_subset["M_ACC"], 1)
                p = np.poly1d(z)
                x_min, x_max = model_subset["HumanValue"].min(), model_subset["HumanValue"].max()
                if x_max > x_min:
                    x_line = np.linspace(x_min, x_max, 100)
                    ax.plot(x_line, p(x_line), "--", color="grey", alpha=0.5, linewidth=1.5)
            except (np.linalg.LinAlgError, ValueError):
                pass  # Skip regression line if fit fails
    
    ax.set_xlabel("", fontsize=16, fontweight="bold")
    ax.set_ylabel("Baseline Model Accuracy" if idx == 0 else "", fontsize=16, fontweight="bold")
    ax.set_title(var_name, fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# Create legend - show datasets (colors) only, models are shown by shapes
handles, labels = axes[0].get_legend_handles_labels()
# Remove duplicate labels
seen = set()
unique_handles = []
unique_labels = []
for h, l in zip(handles, labels):
    if l not in seen and l:
        seen.add(l)
        unique_handles.append(h)
        unique_labels.append(l)
axes[0].legend(unique_handles, unique_labels, title="Dataset", fontsize=12, title_fontsize=14, frameon=True)

plt.tight_layout()
plt.savefig("Fig_5_Baseline_Zero_Shot_Model_Accuracy_vs_Task_Inherent_Errors.png", dpi=300, bbox_inches='tight')
plt.close()

print("Fig_5_Baseline_Zero_Shot_Model_Accuracy_vs_Task_Inherent_Errors.png saved.")

# ============================================================================
# Visualization 5: RQ2 Gain
# ============================================================================

print("Creating Fig_6_Prompting_Strategies_Effectiveness_vs_Task_Inherent_Errors.png...")

# Calculate gains
m_cols = [c for c in df_all.columns if c.startswith("M_")]
baseline = df_all[df_all["Technique"] == "Zero-shot"].copy()
baseline = baseline.groupby(["model", "data"])[m_cols].first().reset_index()
baseline.columns = [c if c in ["model", "data"] else c + "_zero" for c in baseline.columns]

df_all_gain = df_all.merge(baseline, on=["model", "data"], how="left")
for col in m_cols:
    df_all_gain[f"Gain_{col}"] = df_all_gain[col] - df_all_gain[f"{col}_zero"]

# Get top gain per (data, model)
top_gain_ACC = df_all_gain[~df_all_gain["Technique"].str.contains("APO", case=False, na=False)].copy()
top_gain_ACC = top_gain_ACC.loc[top_gain_ACC.groupby(["data", "model"])["Gain_M_ACC"].idxmax()]

# Get baseline metrics
zs_base = df_all_gain[df_all_gain["Technique"] == "Zero-shot"].copy()
zs_base = zs_base[["data", "model", "M_ACC", "M_AS", "AER_task", "CRI_task", "AER_model", "CRI_model"]].copy()
zs_base.columns = ["data", "model"] + [f"base_{c}" for c in zs_base.columns[2:]]

top_gain_ACC = top_gain_ACC.merge(zs_base, on=["data", "model"], how="left")

# Check if H_kripp_alpha has any non-NaN values
has_kripp_alpha_gain = "H_kripp_alpha" in top_gain_ACC.columns and top_gain_ACC["H_kripp_alpha"].notna().any()

# Select variables for plotting (exclude H_kripp_alpha if all NaN)
value_vars_gain = ["H_ACC", "H_AER", "H_CRI",
                   "base_AER_task", "base_CRI_task", "base_AER_model", "base_CRI_model"]
if has_kripp_alpha_gain:
    value_vars_gain.insert(1, "H_kripp_alpha")  # Insert after H_ACC

plot_df = top_gain_ACC.melt(
    id_vars=["data", "model", "Gain_M_ACC"],
    value_vars=value_vars_gain,
    var_name="Variable",
    value_name="HumanValue"
)

plot_df["Variable"] = plot_df["Variable"].map({
    "H_ACC": "H-Accuracy",
    "H_kripp_alpha": "H-Agree",
    "H_AER": "H-Boundary",
    "H_CRI": "H-Conceptual",
    "base_AER_task": "M-Boundary (Task)",
    "base_CRI_task": "M-Conceptual (Task)",
    "base_AER_model": "M-Boundary (Model)",
    "base_CRI_model": "M-Conceptual (Model)"
})

# Build variable order based on what we have
variable_order_gain = ["H-Accuracy"]
if has_kripp_alpha_gain:
    variable_order_gain.append("H-Agree")
variable_order_gain.extend(["H-Boundary", "H-Conceptual",
                            "M-Boundary (Task)", "M-Conceptual (Task)", 
                            "M-Boundary (Model)", "M-Conceptual (Model)"])
plot_df["Variable"] = pd.Categorical(plot_df["Variable"], categories=variable_order_gain, ordered=True)

# Store original data names before mapping for color consistency
plot_df["data_original"] = plot_df["data"].copy()
# Map data to display names
plot_df["data"] = plot_df["data"].map({
    "bloom": "Bloom",
    "gug": "GuG",
    "mathdial": "MathDial",
    "uptake": "Uptake"
})

# Adjust number of subplots based on available variables
n_plots_gain = len(variable_order_gain)
fig, axes = plt.subplots(1, n_plots_gain, figsize=(2.5*n_plots_gain, 4), sharey=True)
if n_plots_gain == 1:
    axes = [axes]  # Make it iterable

# Use consistent dataset order to match R's factor levels (using display names)
dataset_order_display = ["Bloom", "GuG", "MathDial", "Uptake"]

# Use ggplot2's default discrete color palette (Set1 from RColorBrewer)
# Colors: Red, Blue, Green, Purple (matching R's factor order: bloom, gug, mathdial, uptake)
ggplot_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF']
# Create color map using display names in the correct order
color_map = {ds: ggplot_colors[i % len(ggplot_colors)] for i, ds in enumerate(dataset_order_display)}

# Ensure we iterate in the correct variable order
for idx, var_name in enumerate(variable_order_gain):
    var_group = plot_df[plot_df["Variable"] == var_name]
    if len(var_group) == 0:
        continue
    ax = axes[idx]
    
    # Get unique models and datasets for colors and shapes
    models = sorted(var_group["model"].unique())
    # Use display names in the order they should appear
    datasets_ordered = [ds for ds in dataset_order_display if ds in var_group["data"].unique()]
    datasets_ordered.extend([d for d in var_group["data"].unique() if d not in datasets_ordered])
    # Shape map for models (circles and triangles)
    shape_map = {mod: ['o', '^', 's', 'D'][i % 4] for i, mod in enumerate(models)}
    
    # Plot points by dataset (color) and model (shape)
    for data_name in datasets_ordered:
        for model_name in models:
            subset = var_group[(var_group["data"] == data_name) & (var_group["model"] == model_name)].copy()
            # Remove NaN and infinite values
            subset = subset[
                subset["HumanValue"].notna() & 
                subset["Gain_M_ACC"].notna() &
                np.isfinite(subset["HumanValue"]) &
                np.isfinite(subset["Gain_M_ACC"])
            ]
            
            if len(subset) > 0:
                # Use color for dataset, shape for model
                label = data_name if model_name == models[0] else ""  # Only label first model per dataset
                ax.scatter(subset["HumanValue"], subset["Gain_M_ACC"], 
                          c=[color_map[data_name]], marker=shape_map[model_name],
                          label=label, s=100, alpha=0.7, edgecolors='none')
    
    # Fit regression lines by MODEL (not by dataset)
    for model_name in models:
        model_subset = var_group[var_group["model"] == model_name].copy()
        model_subset = model_subset[
            model_subset["HumanValue"].notna() & 
            model_subset["Gain_M_ACC"].notna() &
            np.isfinite(model_subset["HumanValue"]) &
            np.isfinite(model_subset["Gain_M_ACC"])
        ]
        
        if len(model_subset) > 1:
            try:
                z = np.polyfit(model_subset["HumanValue"], model_subset["Gain_M_ACC"], 1)
                p = np.poly1d(z)
                x_min, x_max = model_subset["HumanValue"].min(), model_subset["HumanValue"].max()
                if x_max > x_min:
                    x_line = np.linspace(x_min, x_max, 100)
                    ax.plot(x_line, p(x_line), "--", color="grey", alpha=0.5, linewidth=1.5)
            except (np.linalg.LinAlgError, ValueError):
                pass  # Skip regression line if fit fails
    
    ax.set_xlabel("", fontsize=16, fontweight="bold")
    ax.set_ylabel("Model Accuracy Gain" if idx == 0 else "", fontsize=16, fontweight="bold")
    ax.set_title(var_name, fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# Create legend - show datasets (colors) only, models are shown by shapes
handles, labels = axes[0].get_legend_handles_labels()
# Remove duplicate labels
seen = set()
unique_handles = []
unique_labels = []
for h, l in zip(handles, labels):
    if l not in seen and l:
        seen.add(l)
        unique_handles.append(h)
        unique_labels.append(l)
axes[0].legend(unique_handles, unique_labels, title="Dataset", fontsize=12, title_fontsize=14, frameon=True)

plt.tight_layout()
plt.savefig("Fig_6_Prompting_Strategies_Effectiveness_vs_Task_Inherent_Errors.png", dpi=300, bbox_inches='tight')
plt.close()

print("Fig_6_Prompting_Strategies_Effectiveness_vs_Task_Inherent_Errors.png saved.")
print("\nAll visualizations completed successfully!")

