from run_test import run_test
from run_metrics import run_metrics
import time

if __name__ == "__main__":
    print(__name__)
    ts_test = time.time()
    print("Test time: ", ts_test)
    run_test()
    ts_metric = time.time()
    print("Metric time: ", ts_metric)
    run_metrics()
