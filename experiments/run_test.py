from numpy import load, array, save
from algorithms import notears, causal_discovery_rl, gran_dag, dag_gnn

import os
from networkx import to_numpy_matrix
from cdt.causality.graph import CAM
from cdt.causality.graph import SAM
import cdt
import time
from pandas import DataFrame
from numpy import float32
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
RESULT_DIR = "results"
cdt.SETTINGS.rpath = os.getenv("RSCRIPT_PATH")  # path to your r executable


def run_notears(data: array, output_path: str):
    print("=================")
    print("Running NOTEARS: ", output_path)
    print("=================")
    notear_result_dir = os.path.join(output_path, "notears")
    notear_result_path = os.path.join(notear_result_dir, "result.npy")
    if not os.path.isdir(notear_result_dir):
        os.mkdir(notear_result_dir)
    time0 = time.time()
    output_dict = notears.run(
        notears.notears_standard,
        data,
        notears.loss.least_squares_loss,
        notears.loss.least_squares_loss_grad,
        e=1e-8,
        verbose=False,
    )
    acyclic_W = notears.utils.threshold_output(output_dict["W"])
    time1 = time.time()
    print(time1, time0)
    timing = time1 - time0
    print(timing)
    with open(os.path.join(notear_result_dir, "timing.txt"), "w") as f:
        f.write(str(timing))
    save(notear_result_path, acyclic_W)
    with open(os.path.join(notear_result_dir, "completed.txt"), "w") as f:
        f.write(str("done"))


def run_cam(data: array, output_path: str):
    print("=================")
    print("Running CAM: ", output_path)
    print("=================")
    cam_result_dir = os.path.join(output_path, "cam")
    cam_result_path = os.path.join(cam_result_dir, "result.npy")
    if not os.path.isdir(cam_result_dir):
        os.mkdir(cam_result_dir)
    time0 = time.time()
    obj = CAM()
    output = obj.predict(DataFrame(data).astype(float32))
    pred = to_numpy_matrix(output)
    time1 = time.time()
    print(time1, time0)
    timing = time1 - time0
    print(timing)
    with open(os.path.join(cam_result_dir, "timing.txt"), "w") as f:
        f.write(str(timing))
    save(cam_result_path, pred)
    with open(os.path.join(cam_result_dir, "completed.txt"), "w") as f:
        f.write(str("done"))


def run_sam(data: array, output_path: str):
    print("=================")
    print("Running SAM: ", output_path)
    print("=================")
    sam_result_dir = os.path.join(output_path, "sam")
    sam_result_path = os.path.join(sam_result_dir, "result.npy")
    if not os.path.isdir(sam_result_dir):
        os.mkdir(sam_result_dir)
    time0 = time.time()
    obj = SAM(gpus=1, njobs=2, train_epochs=120, test_epochs=25, nruns=2)
    output = obj.predict(DataFrame(data).astype(float32))
    pred = to_numpy_matrix(output)
    time1 = time.time()
    print(time1, time0)
    timing = time1 - time0
    print(timing)
    with open(os.path.join(sam_result_dir, "timing.txt"), "w") as f:
        f.write(str(timing))
    save(sam_result_path, pred)
    with open(os.path.join(sam_result_dir, "completed.txt"), "w") as f:
        f.write(str("done"))


def run_rl_bic(data_path: str, output_path: str, shape: tuple):
    print("=================")
    print("Running RL_BIC: ", output_path)
    print("=================")
    rl_bic_result_dir = os.path.join(output_path, "rl_bic")
    if not os.path.isdir(rl_bic_result_dir):
        os.mkdir(rl_bic_result_dir)
    causal_discovery_rl.main(
        {
            "data_path": data_path,
            "graph_dir": rl_bic_result_dir,
            "read_data": True,
            "max_length": shape[1],
            "data_size": shape[0],
            "lambda_flag_default": True,
            "nb_epoch": 2000,
        }
    )
    with open(os.path.join(rl_bic_result_dir, "completed.txt"), "w") as f:
        f.write(str("done"))


def run_dag_gnn(data_path: str, output_path: str, shape: tuple):
    print("=================")
    print("Running DAG_GNN: ", output_path)
    print("=================")
    dag_gnn_result_dir = os.path.join(output_path, "dag_gnn")
    if not os.path.isdir(dag_gnn_result_dir):
        os.mkdir(dag_gnn_result_dir)
    dag_gnn.run_dag_gnn(
        {
            "data_path": data_path,
            "model": "NonLinGaussANM",
            "output_path": dag_gnn_result_dir,
            "num_vars": shape[1],
            "data_size": shape[0],
            "train_iter": 2000,
        }
    )
    with open(os.path.join(dag_gnn_result_dir, "completed.txt"), "w") as f:
        f.write(str("done"))


def run_gran_dag(data_path: str, output_path: str, shape: tuple):
    print("=================")
    print("Running GRAN_DAG: ", output_path)
    print("=================")
    gran_dag_result_dir = os.path.join(output_path, "gran_dag")
    if not os.path.isdir(gran_dag_result_dir):
        os.mkdir(gran_dag_result_dir)
    time0 = time.time()
    gran_dag.run_gran_dag(
        {
            "data_path": data_path,
            "model": "NonLinGaussANM",
            "output_path": gran_dag_result_dir,
            "num_vars": shape[1],
            "data_size": shape[0],
            "train_iter": 2000,
        }
    )
    time1 = time.time()
    print(time1, time0)
    timing = time1 - time0
    print(timing)


def run_test():
    causal_mech_dirs = next(os.walk(DATA_DIR))[1]
    for causal_mech_dir in causal_mech_dirs:
        if not os.path.isdir(RESULT_DIR):
            os.mkdir(RESULT_DIR)
        # Configure paths
        causal_mech_data_dir = os.path.join(DATA_DIR, causal_mech_dir)
        dataset_dirs = next(os.walk(causal_mech_data_dir))[1]
        causal_mech_result_dir = os.path.join(RESULT_DIR, causal_mech_dir)
        # Create result dir
        if not os.path.isdir(causal_mech_result_dir):
            os.mkdir(causal_mech_result_dir)
        for dataset in dataset_dirs:
            # Configure paths
            dataset_dir = os.path.join(causal_mech_data_dir, dataset)
            data_path = os.path.join(dataset_dir, "data1.npy")
            dataset_result_dir = os.path.join(causal_mech_result_dir, dataset)
            # Create result dir
            if not os.path.isdir(dataset_result_dir):
                os.mkdir(dataset_result_dir)
            # Load test data
            test_data = load(data_path)

            dag_gnn_result_dir = os.path.join(dataset_result_dir, "dag_gnn")
            gran_dag_result_dir = os.path.join(dataset_result_dir, "gran_dag")
            rl_bic_result_dir = os.path.join(dataset_result_dir, "rl_bic")
            sam_result_dir = os.path.join(dataset_result_dir, "sam")

            # # Run NoTears
            # run_notears(data=test_data, output_path=dataset_result_dir)

            # # Run CAM (https://arxiv.org/abs/1310.1533)
            # run_cam(data=test_data, output_path=dataset_result_dir)

            if not os.path.isdir(sam_result_dir):
                # Run SAM (https://arxiv.org/abs/1803.04929v5)
                run_sam(data=test_data, output_path=dataset_result_dir)
            else:
                if len(os.listdir(sam_result_dir)) == 0:
                    # Run SAM (https://arxiv.org/abs/1803.04929v5)
                    run_sam(data=test_data, output_path=dataset_result_dir)

            if not os.path.isdir(rl_bic_result_dir):
                # Run RL-BIC
                run_rl_bic(
                    data_path=dataset_dir,
                    output_path=dataset_result_dir,
                    shape=test_data.shape,
                )
            else:
                if len(os.listdir(rl_bic_result_dir)) == 0:
                    # Run RL-BIC
                    run_rl_bic(
                        data_path=dataset_dir,
                        output_path=dataset_result_dir,
                        shape=test_data.shape,
                    )

            if not os.path.isdir(gran_dag_result_dir):
                # Run Gran Dag
                run_gran_dag(
                    data_path=dataset_dir,
                    output_path=dataset_result_dir,
                    shape=test_data.shape,
                )
            else:
                if len(os.listdir(gran_dag_result_dir)) == 0:
                    # Run Gran Dag
                    run_gran_dag(
                        data_path=dataset_dir,
                        output_path=dataset_result_dir,
                        shape=test_data.shape,
                    )

            if not os.path.isdir(dag_gnn_result_dir):
                # Run DAG-GNN
                run_dag_gnn(
                    data_path=dataset_dir,
                    output_path=dataset_result_dir,
                    shape=test_data.shape,
                )
            else:
                if len(os.listdir(dag_gnn_result_dir)) == 0:
                    # Run DAG-GNN
                    run_dag_gnn(
                        data_path=dataset_dir,
                        output_path=dataset_result_dir,
                        shape=test_data.shape,
                    )


if __name__ == "__main__":
    print("Running test")
    run_test()
