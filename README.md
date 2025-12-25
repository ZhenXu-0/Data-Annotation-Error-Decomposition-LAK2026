# Enhancing LLM-Based Data Annotation with Error Decomposition

This repository contains data and code needed to replicate the analyses presented in the paper below, as well as code for applying the proposed error decomposition method to new data annotation tasks.

**Paper reference**
Zhen Xu, Vedant Khatri, Diana Dai, Xiner Liu, Siyan, Xuanming, Renzhe Yu. (2025) Enhancing LLM-Based Data Annotation with Error Decomposition. In Proceedings of the 16th International Learning Analytics & Knowledge Conference. (LAK'26)

## Appendix
The appendix of the paper is available here: [Appendix (PDF)](LAK26_Error_Decomposition_appendix.pdf)
# Error Decomposition for Ordinal Classification Tasks

This repository contains Python implementations for analyzing error decomposition in ordinal classification tasks, as described in the paper. The code calculates task-inherent errors (due to human annotation ambiguity) versus model-specific errors.

## Table of Contents

1. [Setup](#setup)
2. [Data Description](#data-description)
3. [Part 1: Replicating Paper Results](#part-1-replicating-paper-results)
4. [Part 2: Applying Error Decomposition to Your Own Task](#part-2-applying-error-decomposition-to-your-own-task)

---

## Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:
   ```bash
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

4. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Description

This section describes the data files used in this repository and their structure.

### Directory Structure

```
Lak_code/
├── human_annotation/          # Human annotation data files
│   ├── Bloom-All_20.csv       # Bloom taxonomy annotations (20 items)
│   ├── Bloom-All_30.csv       # Bloom taxonomy annotations (30 items)
│   ├── Bloom-All_40.csv       # Bloom taxonomy annotations (40 items)
│   ├── bloom_agreement_20.csv # Bloom inter-annotator agreement metrics
│   ├── MathDial-All_20.csv    # MathDial taxonomy annotations
│   ├── mathdial_agreement_20.csv
│   ├── Uptake-All_20.csv      # Uptake taxonomy annotations
│   ├── uptake_agreement_20.csv
│   ├── GUG-All_20.csv         # GUG taxonomy annotations
│   └── gug_agreement_20.csv
├── performance_summary/       # Model performance data files
│   ├── bloom_full.csv         # Full model predictions for Bloom
│   ├── bloom_summary.csv      # Summary statistics for Bloom
│   ├── mathdial_full.csv      # Full model predictions for MathDial
│   ├── uptake_full.csv        # Full model predictions for Uptake
│   └── gug_full.csv           # Full model predictions for GUG
└── error_decomposition_sample_data/  # Sample data for Part 2
    ├── human_annotation.csv   # Example human annotations
    └── model_annotation.csv    # Example model predictions
```

### Human Annotation Files

**Format**: CSV files with wide format (one column per taxonomy level)

**File naming convention**: `{Dataset}-All_{N}.csv` where:
- `{Dataset}`: Dataset name (Bloom, MathDial, Uptake, GUG)
- `{N}`: Number of annotated items (20, 30, or 40)

**Required columns**:
- `Annotator`: Identifier for the annotator (e.g., "groundtruth", "A1", "A2")
- `ID`: Unique identifier for each learning outcome/item
- `Learning_outcome`: Text description of the item (optional, for reference)
- **Taxonomy level columns**: One column per taxonomy level with binary values (1 = selected, empty/0 = not selected)

**Example structure (Bloom taxonomy)**:
```csv
Annotator,ID,Learning_outcome,Remember,Understand,Apply,Analyze,Evaluate,Create
groundtruth,1,Develop a plan for their first internship.,,,,,,1
groundtruth,2,Evaluate design options.,,,,,1,
A1,1,Develop a plan for their first internship.,,,,1,,
A2,1,Develop a plan for their first internship.,,,,,,1
```

**Taxonomy levels by dataset**:
- **Bloom**: Remember, Understand, Apply, Analyze, Evaluate, Create (6 levels)
- **MathDial**: Focus, Probing, Telling, Generic (4 levels)
- **Uptake**: Low, Mid, High (3 levels)
- **GUG**: 1, 2, 3, 4 (4 levels, numeric)

**Agreement files**: Files named `{dataset}_agreement_{N}.csv` contain inter-annotator agreement metrics (Krippendorff's alpha) calculated for each dataset.

### Model Performance Files

**Format**: CSV files with model predictions

**File naming convention**: `{dataset}_full.csv` for detailed predictions, `{dataset}_summary.csv` for aggregated statistics

**Required columns**:
- `outcome_id`: Unique identifier matching the `ID` in human annotation files
- `learning_outcome`: Text description (optional, for reference)
- `human_category`: The human-annotated category (ground truth)
- `model_category`: The model-predicted category
- `exact_match`: Boolean indicating if prediction matches ground truth
- `Technique`: Prompting strategy used (e.g., "Zero-shot", "Few-shot", "Chain-of-Thought")
- `model`: Model name (e.g., "GPT-3.5", "GPT-4")

**Example structure**:
```csv
outcome_id,learning_outcome,human_category,model_category,exact_match,Technique,model
0,"Demonstrate research skills...","apply","analyze",False,"Zero-shot","GPT-3.5"
1,"Identify local, national...","remember","analyze",False,"Zero-shot","GPT-3.5"
2,"Assess space-time coding...","evaluate","analyze",False,"Zero-shot","GPT-3.5"
```

### Sample Data Files (for Part 2)

Located in `error_decomposition_sample_data/`:

- **`human_annotation.csv`**: Example human annotation file using Bloom taxonomy format (20 items, 2 annotators + ground truth)
- **`model_annotation.csv`**: Example model prediction file (150 items) with `human_category` and `model_category` columns

These files serve as templates for users who want to apply error decomposition to their own tasks.

### Data Requirements Summary

For **Part 1** (replicating paper results):
- Human annotation files: `human_annotation/{Dataset}-All_{N}.csv`
- Agreement files: `human_annotation/{dataset}_agreement_{N}.csv`
- Model performance files: `performance_summary/{dataset}_full.csv`

For **Part 2** (your own task):
- Human annotation file: CSV with `Annotator`, `ID`, and taxonomy level columns
- Model annotation file: CSV with `human_category` and `model_category` columns

---

## Part 1: Replicating Paper Results

This section explains how to run `paper_results.py` to generate the 5 figures from the paper.

### Running the Script

1. **Make sure your virtual environment is activated** (see Setup above)

2. **Run the script**:
   ```bash
   python paper_results.py
   ```

3. **Output**: The script will generate 5 PNG files in the current directory:
   - `Fig2_Distribution_of_error_types_observed_in_human_annotations.png`
   - `Fig_3_Comparison_of_Error_Decomposition_Between_Two_Models.png`
   - `Fig_4_Error_Decomposition_Across_Prompting_Strategies.png`
   - `Fig_5_Baseline_Zero_Shot_Model_Accuracy_vs_Task_Inherent_Errors.png`
   - `Fig_6_Prompting_Strategies_Effectiveness_vs_Task_Inherent_Errors.png`

### Adjusting the Annotation Number

The script supports different annotation numbers (20, 30, or 40 annotations per dataset). To change this:

1. **Open `paper_results.py`** in a text editor

2. **Find the configuration section** at the top of the file (around line 25-28):
   ```python
   # ============================================================================
   # Configuration: Annotation Number
   # ============================================================================
   # Change this value to use different annotation files: 20, 30, or 40
   ANNOTATION_NUMBER = 20  # Options: 20, 30, or 40
   ```

3. **Change the value** to your desired annotation number:
   ```python
   ANNOTATION_NUMBER = 30  # or 40
   ```

4. **Save the file** and run the script again

**Note**: The script will automatically load the corresponding annotation files:
   - `human_annotation/Bloom-All_{ANNOTATION_NUMBER}.csv`
   - `human_annotation/MathDial-All_{ANNOTATION_NUMBER}.csv`
   - `human_annotation/Uptake-All_{ANNOTATION_NUMBER}.csv`
   - `human_annotation/GUG-All_{ANNOTATION_NUMBER}.csv`

**Important**: Currently, the agreement files (`bloom_agreement_20.csv`, etc.) and GUG annotation file are hardcoded to use `_20.csv`. If you need to use different annotation numbers for these, you will need to manually update the file paths in the code (around lines 262, 521-524).

---

## Part 2: Applying Error Decomposition to Your Own Task

This section explains how to use `calculate_error_decomposition.py` to calculate error decomposition percentages for your own ordinal classification task.

### Overview

The script calculates 5 error decomposition percentages that sum to 1.0:
1. **Correct Annotation**: Percentage of exact matches between model and human annotations
2. **Boundary Ambiguity Error (Task)**: Task-inherent errors due to boundary ambiguity
3. **Conceptual Misidentification Error (Task)**: Task-inherent errors due to conceptual misidentification
4. **Boundary Ambiguity Error (Model)**: Model-specific errors due to boundary ambiguity
5. **Conceptual Misidentification Error (Model)**: Model-specific errors due to conceptual misidentification

### Step-by-Step Instructions

#### Step 1: Prepare Your Data Files

You need two CSV files:

**A. Human Annotation File** (`human_annotation.csv`):
- **Required columns**:
  - `Annotator`: Name/ID of the annotator (e.g., "A1", "A2", "groundtruth")
  - `ID`: Unique identifier for each item being annotated
  - **Taxonomy level columns**: One column for each level in your taxonomy (see examples below)

- **Data format options**:
  - **Wide format** (recommended): One column per taxonomy level, with values of 1 (selected) or 0 (not selected)
    ```
    Annotator,ID,Level1,Level2,Level3,Level4
    groundtruth,1,0,0,1,0
    A1,1,0,1,0,0
    A2,1,0,0,1,0
    ```
  - **Long format**: One row per annotation with a `Label` column
    ```
    Annotator,ID,Label
    groundtruth,1,Level3
    A1,1,Level2
    A2,1,Level3
    ```

**B. Model Annotation File** (`model_annotation.csv`):
- **Required columns**:
  - `human_category`: The human-annotated category for each item
  - `model_category`: The model-predicted category for each item
- **Optional columns**: Any additional metadata (model name, technique, etc.) - these are ignored by the script

- **Example**:
  ```
  outcome_id,human_category,model_category
  1,analyze,analyze
  2,remember,analyze
  3,evaluate,evaluate
  ```

#### Step 2: Configure the Script

1. **Open `calculate_error_decomposition.py`** in a text editor

2. **Update the file paths** (around lines 29-30):
   ```python
   HUMAN_ANNOTATION_FILE = "path/to/your/human_annotation.csv"
   MODEL_ANNOTATION_FILE = "path/to/your/model_annotation.csv"
   ```

3. **Set the number of annotations** (around lines 37-38):
   ```python
   NUM_HUMAN_ANNOTATIONS = 20  # Change to match your human annotation count
   NUM_MODEL_ANNOTATIONS = 150  # Change to match your model annotation count
   ```
   *Note: These are for reference/info only and don't affect calculations*

4. **Define your taxonomy levels** (around line 43):
   ```python
   TAXONOMY_LEVELS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
   ```
   
   **Important**: 
   - List levels in order from **lowest to highest** (this defines the ordinal structure)
   - Use **lowercase** names (the script normalizes all labels to lowercase)
   - The script will automatically calculate distances between levels based on this ordering
   
   **Examples for different tasks**:
   ```python
   # MathDial (4 levels):
   TAXONOMY_LEVELS = ["focus", "probing", "telling", "generic"]
   
   # Uptake (3 levels):
   TAXONOMY_LEVELS = ["low", "mid", "high"]
   
   # GUG (4 levels, numeric):
   TAXONOMY_LEVELS = ["1", "2", "3", "4"]
   
   # Custom task with 5 levels:
   TAXONOMY_LEVELS = ["beginner", "intermediate", "advanced", "expert", "master"]
   ```

5. **Update column names** (if needed):
   - The script expects `Annotator` and `ID` columns in the human annotation file
   - The script expects `human_category` and `model_category` columns in the model annotation file
   - If your columns have different names, you'll need to modify the code (see comments around line 54-57)

#### Step 3: Run the Script

1. **Make sure your virtual environment is activated** (see Setup above)

2. **Run the script**:
   ```bash
   python calculate_error_decomposition.py
   ```

3. **Output**: The script will print the 5 error decomposition percentages to the console:
   ```
   ============================================================
   ERROR DECOMPOSITION RESULTS
   ============================================================
   
   The 5 percentages (sum to 1.0):
   ============================================================
   Correct Annotation                           :  50.00% (0.5000)
   Boundary Ambiguity Error (Task)              :   9.35% (0.0935)
   Conceptual Misidentification Error (Task)    :   2.33% (0.0233)
   Boundary Ambiguity Error (Model)             :   6.65% (0.0665)
   Conceptual Misidentification Error (Model)   :  31.67% (0.3167)
   ============================================================
   Total                                        : 100.00% (1.0000)
   ============================================================
   ```

### Example: Using Sample Data

The repository includes sample data files in the `error_decomposition_sample_data/` folder:
- `human_annotation.csv`: Example human annotations (Bloom taxonomy, 20 items)
- `model_annotation.csv`: Example model predictions (150 items)

To test the script with these sample files, the configuration is already set up. Just run:
```bash
python calculate_error_decomposition.py
```

### Troubleshooting

**Issue**: "FileNotFoundError" when running the script
- **Solution**: Check that the file paths in the configuration section are correct and relative to the script location

**Issue**: "KeyError" for column names
- **Solution**: Verify that your CSV files have the required columns (`Annotator`, `ID` for human data; `human_category`, `model_category` for model data)

**Issue**: Taxonomy levels not matching
- **Solution**: Ensure that:
  1. Your `TAXONOMY_LEVELS` list matches the actual category names in your data (case-insensitive)
  2. All categories in your data files are valid taxonomy levels
  3. The script normalizes all labels to lowercase, so "Analyze" and "analyze" are treated the same

**Issue**: Results don't sum to 1.0
- **Solution**: This should not happen. If it does, check that:
  1. All categories in your data are valid taxonomy levels
  2. There are no missing values in critical columns
  3. The data files are properly formatted

### Understanding the Results

- **Correct Annotation**: Higher is better. This is the percentage of items where the model prediction exactly matches the human annotation.

- **Task-Inherent Errors** (Boundary Ambiguity and Conceptual Misidentification): These represent errors that are "built into" the task due to human annotation ambiguity. Even if a model were perfect, it might still make these errors because humans disagree on the correct label.

- **Model-Specific Errors**: These represent errors that are specific to the model's performance, beyond what can be explained by human annotation ambiguity.

The decomposition helps you understand:
- How much of the model's error is due to task ambiguity vs. model limitations
- Whether improving the model would help, or if the task itself is inherently ambiguous

---

## Requirements

See `requirements.txt` for the complete list of Python packages. Main dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- scipy

---

## Deactivating Virtual Environment

When you're done working:
```bash
deactivate
```

---

## Citation

If you use this code in your research, please cite the original paper.

---

## License

[Add your license information here]

---

## Contact

[Add contact information here]
