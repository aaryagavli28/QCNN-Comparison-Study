# run_all_experiments.py

import json
import pandas as pd

from experiment_runner import run_experiment

from models.classical import Net as ClassicalNet
from models.single_encoding import Net as SingleNet
from models.multi_encoding import Net as MultiNet
from models.hybrid_layer import Net as HybridNet
from models.inception import Net as InceptionNet
from models.multi_noisy import Net as MultiNoisyNet


# -------------------------
# DATASETS
# -------------------------

DATASETS = {
    "mnist_179":"./datasets/mnist_179_1200.csv"
}


# -------------------------
# MODELS
# -------------------------
'''
MODELS = {
    "single_encoding": SingleNet,
    "multi_encoding": MultiNet,
    "inception": InceptionNet,
    "multi_noisy": MultiNoisyNet
}
'''
MODELS = {
    "inception": InceptionNet
}

# -------------------------
# EXPERIMENT LOOP
# -------------------------

results = []
confusions = {}

for dataset_name, dataset_path in DATASETS.items():

    confusions[dataset_name] = {}

    for model_name, model_class in MODELS.items():

        print(f"Running {dataset_name} - {model_name}")

        try:

            result = run_experiment(
                dataset_path,
                model_class,
                model_name
            )

            results.append(result)

            confusions[dataset_name][model_name] = \
                result["confusion_matrix"]

            # -------------------------
            # IMMEDIATE SAVE
            # -------------------------

            pd.DataFrame(results).to_csv(
                "metrics.csv",
                index=False
            )

            with open(
                "confusion_matrices.json",
                "w"
            ) as f:

                json.dump(
                    confusions,
                    f,
                    indent=4
                )

            print(
                f"Saved results for {dataset_name} - {model_name}"
            )

        except Exception as e:

            print(f"FAILED {dataset_name} {model_name}")
            print(e)


# -------------------------
# FINAL SAVE
# -------------------------

df = pd.DataFrame(results)

df.to_csv(
    "metrics.csv",
    index=False
)

with open(
    "confusion_matrices.json",
    "w"
) as f:

    json.dump(
        confusions,
        f,
        indent=4
    )

print("Finished all experiments.")