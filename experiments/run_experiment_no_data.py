from generate_data import generate_data
from run_test import run_test
from run_metrics import run_metrics

if __name__ == "__main__":
    print(__name__)
    generate_data()
    run_test()
    run_metrics()
