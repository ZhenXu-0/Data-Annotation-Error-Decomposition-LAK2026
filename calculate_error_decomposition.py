"""
Error Decomposition Calculator

This script calculates the error decomposition percentages for model predictions
compared to human annotations. It outputs 5 percentages that sum to 1:
1. Correct Annotation
2. Boundary Ambiguity Error (Task)
3. Conceptual Misidentification Error (Task)
4. Boundary Ambiguity Error (Model)
5. Conceptual Misidentification Error (Model)

HOW TO USE WITH YOUR OWN DATA:
1. Update the file paths in the CONFIGURATION section below
2. Adjust the taxonomy levels if your data uses different categories
3. Ensure your data files have the required columns (see comments in code)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - Modify these for your own data
# ============================================================================

# File paths - UPDATE THESE FOR YOUR DATA
# Sample data files in error_decomposition_sample_data folder
HUMAN_ANNOTATION_FILE = "error_decomposition_sample_data/human_annotation.csv"
MODEL_ANNOTATION_FILE = "error_decomposition_sample_data/model_annotation.csv"

# ============================================================================
# Task Configuration - UPDATE THESE FOR YOUR DATA
# ============================================================================

# Number of annotations in your dataset (for reference/info only)
NUM_HUMAN_ANNOTATIONS = 20  # Change to match your human annotation count
NUM_MODEL_ANNOTATIONS = 150  # Change to match your model annotation count

# Taxonomy levels - UPDATE BASED ON YOUR TASK
# Define your taxonomy levels here (in order from lowest to highest level)
# This example uses Bloom taxonomy with 6 levels
TAXONOMY_LEVELS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]

# Examples for other tasks:
# MathDial (4 levels): TAXONOMY_LEVELS = ["focus", "probing", "telling", "generic"]
# Uptake (3 levels): TAXONOMY_LEVELS = ["low", "mid", "high"]
# GUG (4 levels): TAXONOMY_LEVELS = ["1", "2", "3", "4"]  # (numeric as strings)
# Custom task: TAXONOMY_LEVELS = ["level1", "level2", "level3", ...]  # Add as many levels as needed
# 
# IMPORTANT: All taxonomy level names should be defined ONLY in TAXONOMY_LEVELS above.
# The code will automatically use these levels for all calculations.

# Column names in your files - UPDATE IF YOUR COLUMNS HAVE DIFFERENT NAMES
# Human annotation file should have columns: Annotator, ID, and taxonomy level columns
# Model annotation file should have columns: human_category, model_category
# (Note: model/technique columns are NOT required - they're only used for labeling)

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
    This calculates the graph distance between taxonomy levels
    """
    n = len(levels)
    d = np.full((n, n), np.inf)
    np.fill_diagonal(d, 0)
    
    # Set adjacency edges
    for e in edges:
        i, j = levels.index(e[0]), levels.index(e[1])
        d[i, j] = 1
        d[j, i] = 1
    
    # Floyd-Warshall algorithm
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

# ============================================================================
# Load and Process Human Annotation Data
# ============================================================================

print("Loading human annotation data...")
print(f"File: {HUMAN_ANNOTATION_FILE}")
print(f"Expected number of human annotations: {NUM_HUMAN_ANNOTATIONS}")

# Load human annotation data
df_human = pd.read_csv(HUMAN_ANNOTATION_FILE)

# Detect data format: wide format (columns for each taxonomy level) or long format
# Check if any taxonomy level names (case-insensitive) exist as columns
taxonomy_cols = []
for level in TAXONOMY_LEVELS:
    # Check for exact match, capitalized, or lowercase
    possible_names = [level, level.capitalize(), level.title(), level.upper()]
    for name in possible_names:
        if name in df_human.columns:
            taxonomy_cols.append(name)
            break

if taxonomy_cols:
    # Wide format: pivot to long format
    df_human_long = df_human.melt(
        id_vars=[c for c in df_human.columns if c not in taxonomy_cols],
        value_vars=taxonomy_cols,
        var_name="Label",
        value_name="Value"
    )
    df_human_long = df_human_long[df_human_long["Value"] == 1].drop(columns=["Value"])
    df_human_long["ID"] = pd.to_numeric(df_human_long["ID"])
else:
    # Already in long format - adjust column names as needed
    df_human_long = df_human.copy()
    if "Label" not in df_human_long.columns:
        # Try to find the label column
        label_cols = [c for c in df_human_long.columns if "label" in c.lower() or "category" in c.lower()]
        if label_cols:
            df_human_long = df_human_long.rename(columns={label_cols[0]: "Label"})

# Build distance matrix for taxonomy
move_levels = [l.lower() for l in TAXONOMY_LEVELS]
adj_edges = [[move_levels[i], move_levels[i+1]] for i in range(len(move_levels)-1)]
dmat = make_dmat(move_levels, adj_edges)

# Process human data
dat_human = df_human_long[["Annotator", "ID", "Label"]].copy()
dat_human["Annotator"] = dat_human["Annotator"].astype(str)
dat_human["ID"] = dat_human["ID"].astype(int)
dat_human["Label"] = dat_human["Label"].apply(normalize)
dat_human = dat_human.drop_duplicates(subset=["Annotator", "ID"], keep='first')

# Get ground truth (human consensus or gold standard)
gt_names = ["groundtruth", "ground_truth", "gold", "gt"]
gt_human = dat_human[dat_human["Annotator"].str.lower().isin(gt_names)][["ID", "Label"]].rename(columns={"Label": "gt"})

# Get human annotator predictions (excluding ground truth)
pred_human = dat_human[~dat_human["Annotator"].str.lower().isin(gt_names)].copy()
pred_human["Annotator"] = pred_human["Annotator"].replace(["A1", "A2"], "A_combined")

# Calculate human metrics
K = len(move_levels)
denom = K - 1 if K > 1 else 1

pred_gt_human = pred_human.merge(gt_human, on="ID", how="left")
pred_gt_human = pred_gt_human[
    pred_gt_human["gt"].notna() &
    pred_gt_human["Label"].isin(move_levels) &
    pred_gt_human["gt"].isin(move_levels)
].copy()

pred_gt_human["delta"] = pred_gt_human.apply(
    lambda row: dmat.loc[row["gt"], row["Label"]], axis=1
)
pred_gt_human["is_match"] = pred_gt_human["delta"] == 0
pred_gt_human["is_adj"] = pred_gt_human["delta"] == 1
pred_gt_human["is_cross"] = pred_gt_human["delta"] >= 2

# Aggregate human metrics - calculate overall metrics across all annotations
# (not averaging per-annotator means, but calculating overall mean)
H_ACC = pred_gt_human["is_match"].mean()
H_AER = pred_gt_human["is_adj"].mean()
H_CRI = pred_gt_human["is_cross"].mean()

print(f"Human Metrics:")
print(f"  Accuracy (H_ACC): {H_ACC:.4f}")
print(f"  Adjacent Error Rate (H_AER): {H_AER:.4f}")
print(f"  Cross-Level Risk (H_CRI): {H_CRI:.4f}")

# ============================================================================
# Load and Process Model Annotation Data
# ============================================================================

print(f"\nLoading model annotation data...")
print(f"File: {MODEL_ANNOTATION_FILE}")
print(f"Expected number of model annotations: {NUM_MODEL_ANNOTATIONS}")
print(f"Taxonomy levels ({len(TAXONOMY_LEVELS)} levels): {', '.join(TAXONOMY_LEVELS)}")

# Load model data
# NOTE: The CSV file should contain model predictions with human_category and model_category columns
df_model = pd.read_csv(MODEL_ANNOTATION_FILE)

# Normalize categories
df_model["human_category"] = df_model["human_category"].astype(str).apply(normalize)
df_model["model_category"] = df_model["model_category"].astype(str).apply(normalize)

# Filter to valid taxonomy levels
df_model = df_model[
    df_model["human_category"].isin(move_levels) &
    df_model["model_category"].isin(move_levels)
].copy()

# Calculate model metrics
df_model["delta"] = df_model.apply(
    lambda row: dmat.loc[row["human_category"], row["model_category"]], axis=1
)
df_model["is_match"] = df_model["delta"] == 0
df_model["is_adj"] = df_model["delta"] == 1
df_model["is_cross"] = df_model["delta"] >= 2

# Aggregate model metrics
M_ACC = df_model["is_match"].mean()
M_AER = df_model["is_adj"].mean()
M_CRI = df_model["is_cross"].mean()

# ============================================================================
# Calculate Error Decomposition
# ============================================================================

print(f"\nCalculating error decomposition...")

# Calculate task vs model error decomposition
AER_task = (H_AER / (M_AER + H_AER + 1e-10)) * M_AER
CRI_task = (H_CRI / (M_CRI + H_CRI + 1e-10)) * M_CRI
AER_model = M_AER - AER_task
CRI_model = M_CRI - CRI_task

# ============================================================================
# Output Results
# ============================================================================

print(f"\n{'='*60}")
print(f"ERROR DECOMPOSITION RESULTS")
print(f"{'='*60}")
print(f"\nThe 5 percentages (sum to 1.0):")
print(f"{'='*60}")

results = {
    "Correct Annotation": M_ACC,
    "Boundary Ambiguity Error (Task)": AER_task,
    "Conceptual Misidentification Error (Task)": CRI_task,
    "Boundary Ambiguity Error (Model)": AER_model,
    "Conceptual Misidentification Error (Model)": CRI_model
}

total = sum(results.values())
for category, value in results.items():
    percentage = value * 100
    print(f"{category:45s}: {percentage:6.2f}% ({value:.4f})")

print(f"{'='*60}")
print(f"{'Total':45s}: {total*100:6.2f}% ({total:.4f})")
print(f"{'='*60}")

# Results are only printed to console, not saved to CSV

print(f"\nDone!")

