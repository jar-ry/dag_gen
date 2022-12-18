from numpy import load
import json
import os
import pathlib
import cdt
from pandas import DataFrame, concat
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = "data"
RESULT_DIR = "results"
cdt.SETTINGS.rpath = os.getenv("RSCRIPT_PATH")  # path to your r executable


def compute_metrics(B_true, B) -> tuple:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        B_true: ground truth graph
        B: predicted graph
    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive

    Codes are from NOTEARS authors.
    """
    d = B.shape[0]
    # linear index of nonzeros

    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    true_size = len(cond)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    acc_res = {
        "fdr": fdr,
        "tpr": tpr,
        "fpr": fpr,
        "shd": shd,
        "false_pos": len(false_pos),
        "false_neg": len(missing_lower),
        "reserved_edges": len(reverse),
        "pred_size": pred_size,
        "true_size": true_size,
    }

    return acc_res


def run_metrics():
    causal_mech_dirs = next(os.walk(RESULT_DIR))[1]
    appended_data = []
    for causal_mech_dir in causal_mech_dirs:
        # Configure paths
        causal_mech_data_dir = os.path.join(DATA_DIR, causal_mech_dir)

        causal_mech_result_dir = os.path.join(RESULT_DIR, causal_mech_dir)
        dataset_result_dirs = next(os.walk(causal_mech_result_dir))[1]
        for dataset_result_dir in dataset_result_dirs:
            # Configure paths
            dataset_dir = os.path.join(causal_mech_data_dir, dataset_result_dir)
            true_dag_path = os.path.join(dataset_dir, "DAG1.npy")

            dataset_result_dir = os.path.join(
                causal_mech_result_dir, dataset_result_dir
            )
            algo_result_dirs = next(os.walk(dataset_result_dir))[1]
            for algo_result_dir in algo_result_dirs:
                result_dir = os.path.join(dataset_result_dir, algo_result_dir)
                result_path = os.path.join(result_dir, "result.npy")
                metric_path = os.path.join(result_dir, "metric.txt")
                timing_path = os.path.join(result_dir, "timing.txt")

                # Create result dir
                true_dag = load(true_dag_path)
                pred_dag = load(result_path)

                # Compute SHD
                metrics = compute_metrics(true_dag, pred_dag)
                # Read compute time
                with open(timing_path) as f:
                    metrics.update({"timing": f.read()})

                with open(metric_path, "w") as f:
                    json.dump(metrics, f)

                metrics["causal_mech"] = causal_mech_dir
                path = pathlib.PurePath(dataset_result_dir)
                metrics["dataset"] = path.name
                metrics["dataset_name"] = "_".join(path.name.split("_")[:-1])
                metrics["dataset_index"] = path.name.split("_")[-1]
                metrics["algo"] = algo_result_dir

                df = DataFrame(metrics, index=[0])
                appended_data.append(df)
    appended_data_df = concat(appended_data)
    appended_data_df.to_csv(os.path.basename(p=os.getcwd()) + "_result.csv")


if __name__ == "__main__":
    print("Running metrics")
    run_metrics()
