from dag_gen.acyclic_graph_generator import AcyclicGraphGenerator
import json
import os

DATA_DIR = "data"


def generate_data():
    with open("data_config.json", "r") as f:
        data_config = json.load(f)
        for dataset_group, datasets in data_config.items():
            dataset_group_path = os.path.join(DATA_DIR, dataset_group)
            if not os.path.isdir(DATA_DIR):
                os.mkdir(DATA_DIR)
            if not os.path.isdir(dataset_group_path):
                os.mkdir(dataset_group_path)
            for dataset_name, generator_config in datasets.items():
                dataset_path = os.path.join(dataset_group_path, dataset_name)
                # If directory doesn't exist or isn't empty
                if not os.path.isdir(dataset_path):
                    os.mkdir(dataset_path)
                if len(os.listdir(dataset_path)) == 0:
                    print("Generating: ", dataset_path)
                    generator = AcyclicGraphGenerator(**generator_config)
                    generator.generate_to_folder(data_path=dataset_path, data_index=1)
                else:
                    print("Skipping: ", dataset_path)

if __name__ == "__main__":
    print("Generating data")
    generate_data()
