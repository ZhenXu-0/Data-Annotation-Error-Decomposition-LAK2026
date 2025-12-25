# Enhancing LLM-Based Data Annotation with Error Decomposition

This repository contains data and code needed to replicate the analyses presented in the paper below, as well as code for applying the proposed error decomposition method to new data annotation tasks.

**Paper reference**
Zhen Xu, Vedant Khatri, Diana Dai, Xiner Liu, Siyan, Xuanming, Renzhe Yu. (2025) Enhancing LLM-Based Data Annotation with Error Decomposition. In Proceedings of the 16th International Learning Analytics & Knowledge Conference. (LAK'26)

## Appendix
The appendix of the paper is available here: [Appendix (PDF)](LAK26_Error_Decomposition_appendix.pdf)

## Table of Contents

1. [Data Description](#data-description)
2. [Setup](#setup)
4. [Replicating Paper Results](#replicating-paper-results)
5. [Applying Error Decomposition to Your Own Task](#applying-error-decomposition-to-your-own-task)

---
## Data Description

### Dataset Structure
```
├── human_annotation/                 # Human annotation data files (for replicating paper results)
│   ├── Bloom-All_20.csv                # Bloom taxonomy annotations (20 items)
│   ├── Bloom-All_30.csv                # Bloom taxonomy annotations (30 items)
│   ├── Bloom-All_40.csv                # Bloom taxonomy annotations (40 items)
│   ├── MathDial-All_20.csv             # MathDial taxonomy annotations (20 items)
│   ├── MathDial-All_30.csv             # MathDial taxonomy annotations (30 items)
│   ├── MathDial-All_40.csv             # MathDial taxonomy annotations (40 items)
│   ├── Uptake-All_20.csv               # Uptake taxonomy annotations (20 items)
│   ├── Uptake-All_30.csv               # Uptake taxonomy annotations (30 items)
│   ├── Uptake-All_40.csv               # Uptake taxonomy annotations (40 items)
│   ├── GUG-All_20.csv                  # GUG taxonomy annotations (20 items)
│   ├── GUG-All_30.csv                  # GUG taxonomy annotations (30 items)
│   └── GUG-All_40.csv                  # GUG taxonomy annotations (40 items)
├── model_annotation/                 # Model annotation data files (for replicating paper results)
│   ├── bloom_full.csv                  # Full model predictions for Bloom
│   ├── gug_full.csv                    # Full model predictions for GUG
│   ├── mathdial_full.csv               # Full model predictions for MathDial
│   ├── uptake_full.csv                 # Full model predictions for Uptake
│   ├── bloom_summary.csv               # Summary statistics for Bloom
│   ├── gug_summary.csv                 # Summary statistics for GUG
│   ├── mathdial_summary.csv            # Summary statistics for MathDial
│   └── uptake_summary.csv              # Summary statistics for Uptake
└── error_decomposition_sample_data/  # Sample data (for applying error decomposition to a new data annotation task)
    ├── human_annotation.csv            # Example human annotations
    └── model_annotation.csv            # Example model predictions
```

### Data Preparation for applying error decomposition to their own tasks
We provide sample data templates in error_decomposition_sample_data/ for applying the error decomposition method to new annotation tasks:

- **`human_annotation.csv`**: Example human annotation file using Bloom taxonomy format (20 items, 2 annotators + ground truth)
- **`model_annotation.csv`**: Example model prediction file (150 human annotation + model annotation)


## Setup

### Prerequisites

- Python 3.7 or higher

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

## Replicating Paper Results

This section illustrates the generation of analysis and results (5 figures) from the paper.

### Running the Script

1. **Run the script**:
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

---

## Applying Error Decomposition to Your Own Task

This section explains how to calculate error decomposition percentages for your own ordinal annotation task.

### Overview

The script calculates 5 error decomposition percentages that sum to 1.0:
1. **Correct Annotation**: Percentage of exact matches between model and human annotations
2. **Boundary Ambiguity Error (Task)**: Task-inherent errors due to boundary ambiguity
3. **Conceptual Misidentification Error (Task)**: Task-inherent errors due to conceptual misidentification
4. **Boundary Ambiguity Error (Model)**: Model-specific errors due to boundary ambiguity
5. **Conceptual Misidentification Error (Model)**: Model-specific errors due to conceptual misidentification

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
  - `human_category`: The human-annotated category (groundtruth) for each item
  - `model_category`: The model-predicted category for each item

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


#### Step 3: Run the Script

1. **Run the script**:
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
---

## Citation

If you use this code in your research, please cite the original paper.

---

## License

[Add your license information here]

---

## Contact

[Add contact information here]
