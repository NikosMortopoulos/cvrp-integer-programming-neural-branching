# CVRP Learned Branching Project

This project contains a custom Branch-and-Bound solver for CVRP MTZ formulation and learned branching policies using GNN and XGBoost.

The basic workflow is:

1. Collect strong-branching samples.
2. Train the GNN policy.
3. Train the XGBoost policy.
4. Run the final Branch-and-Bound experiments.

---

## 1. Setup

Create the environment:

```bash
conda create -n ml4mip python=3.10
conda activate ml4mip
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Check that Gurobi works:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

---

## 2. Collect Branching Samples

Run the data collection script:

```bash
python b_n_b_data_collection.py
```

This script collects strong-branching samples from generated CVRP instances.

Inside the script, data is collected by calling `generate_cvrp_branching_dataset(...)`.

To collect more data, add more calls to this function with different values for:

```text
num_instances
n_customers
q_ratio
samples_per_instance
seed
out_dir
category_type
```

For example, use different values of `n_customers` and `q_ratio` to collect samples from several problem sizes and difficulty levels.

The collected samples are saved in the output directory defined in the script, usually under:

```text
data_arc_features/samples/cvrp/
```

---

## 3. Train the GNN Policy

After collecting samples, run:

```bash
python train_gnn_cvrp.py
```

This trains the GNN to imitate strong branching decisions.

Before running, check the paths inside the training script, especially the sample directory and model output directory.

The trained GNN model is saved under the configured model folder, usually:

```text
models/cvrp_gnn/
```

---

## 4. Train the XGBoost Policy

Run the XGBoost training script:

```bash
python train_xgboost_ranker.py
```

If your script has a different filename, run the file that trains the XGBoost branching ranker.

Before running, check that it reads the same collected sample directory and saves the trained model to the expected output path.

The trained XGBoost model is usually saved under:

```text
outputs/xgboost_ranker_branching/
```

---

## 5. Configure the Final Experiments

The main experiment script is:

```bash
python bnb_neural.py
```

Before running it, configure the problem set inside the file.

The main settings are:

```text
difficulty_categories
PROBLEMS_PER_CATEGORY
BASE_SEED
```

Use `difficulty_categories` to choose the problem sizes and capacity ratios you want to test.

Use `PROBLEMS_PER_CATEGORY` to choose how many problems to run per category.

The script automatically generates the problems, saves them, runs the selected methods, and writes the results to log files.



---

## 6. Choose Which Methods to Run

Inside `bnb_neural.py`, enable or disable the experiment blocks you want.

Common options are:

```text
Gurobi reference
Enhanced B&B without learned branching
Full strong branching
Hybrid GNN + strong branching
Pure GNN branching
XGBoost branching
```

Only leave enabled the methods you want to compare.

---

## 7. Run the Final Experiments

Run:

```bash
python bnb_neural.py
```

The script will run the selected methods on the configured problem categories.

Results are printed in the terminal and saved under:

```text
experiment_logs/
```

Saved problem instances are stored under:

```text
experiment_logs/saved_problems/
```

---

## Minimal Run Order

```bash
python b_n_b_data_collection.py
python train_gnn_cvrp.py
python train_xgboost_ranker.py
python bnb_neural.py
```

If the samples and trained models already exist, run only:

```bash
python bnb_neural.py
```
