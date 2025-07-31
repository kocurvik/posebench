from pathlib import Path
import datetime
import os
import zipfile

import posebench.absolute_pose
import posebench.relative_pose
import posebench.homography
import argparse
from posebench.utils.misc import download_file_with_progress, print_metrics_per_method_table, compute_average_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_iterations", required=False, type=int)
    parser.add_argument("--max_iterations", required=False, type=int)
    parser.add_argument("--success_prob", required=False, type=float)
    parser.add_argument("--method", required=False, type=str)
    parser.add_argument("--dataset", required=False, type=str)
    parser.add_argument("--subsample", required=False, type=int)
    args = parser.parse_args()
    force_opt = {}
    if args.min_iterations is not None:
        force_opt["min_iterations"] = args.min_iterations
    if args.max_iterations is not None:
        force_opt["max_iterations"] = args.max_iterations
    if args.success_prob is not None:
        force_opt["success_prob"] = args.success_prob

    method_filter = []
    if args.method is not None:
        method_filter = args.method.split(",")
    dataset_filter = []
    if args.dataset is not None:
        dataset_filter = args.dataset.split(",")
    return force_opt, method_filter, dataset_filter, args.subsample




def main():
    if not Path("data").is_dir():
        if not Path("data.zip").is_file():
            download_file_with_progress(
                "https://github.com/Parskatt/storage/releases/download/posebench-v0.0.1/data.zip",
                "data.zip",
            )
        print("Extracting data...")
        with zipfile.ZipFile("data.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove("data.zip")
    force_opt, method_filter, dataset_filter, subsample = parse_args()
    problems = {
        "absolute pose": posebench.absolute_pose.main,
        "relative pose": posebench.relative_pose.main,
        "homography": posebench.homography.main,
    }
    start_time = datetime.datetime.now()
    compiled_metrics = []
    dataset_names = []
    for name, problem in problems.items():
        print(f"Running problem {name}")
        metrics, _ = problem(
            force_opt=force_opt,
            method_filter=method_filter,
            dataset_filter=dataset_filter,
            subsample=subsample,
        )

        avg_metrics = compute_average_metrics(metrics)
        compiled_metrics.append(avg_metrics)
        dataset_names += metrics.keys()

    end_time = datetime.datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(
        f"Finished running evaluation in {total_time:.1f} seconds ({len(dataset_names)} datasets)"
    )
    print("Datasets: " + (",".join(dataset_names)) + "\n")

    # Output all the average metrics
    for avg_metrics in compiled_metrics:
        print_metrics_per_method_table(avg_metrics)
        print("")


if __name__ == "__main__":
    main()
