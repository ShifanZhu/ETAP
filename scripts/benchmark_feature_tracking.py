import argparse
from pathlib import Path
import os
import sys
from enum import Enum
import csv

import matplotlib.pyplot as plt

import numpy as np
from prettytable import PrettyTable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import compute_tracking_errors, read_txt_results, SUPPORTED_SEQUENCES_FEATURE_TRACKING

def calculate_mean_by_dataset_type(table, dataset_type):
    values = []
    for row in table.rows:
        if dataset_type == EvalDatasetType.EDS:
            if row[0] in SUPPORTED_SEQUENCES_FEATURE_TRACKING['eds']:
                values.append(float(row[1]))
        else:  # EvalDatasetType.EC
            if row[0] in SUPPORTED_SEQUENCES_FEATURE_TRACKING['ec']:
                values.append(float(row[1]))
    return np.mean(values) if values else 0

def create_summary_csv(tables, output_path):
    header = ['fa_5_eds_mean', 'fa_5_ec_mean', 'te_5_eds_mean', 'te_5_ec_mean', 'inliers_eds_mean', 'inliers_ec_mean']
    data = {}

    for k in tables.keys():
        eds_mean = calculate_mean_by_dataset_type(tables[k], EvalDatasetType.EDS)
        ec_mean = calculate_mean_by_dataset_type(tables[k], EvalDatasetType.EC)
        
        if k.startswith('age_5'):
            data['fa_5_eds_mean'] = eds_mean
            data['fa_5_ec_mean'] = ec_mean
        elif k.startswith('te_5'):
            data['te_5_eds_mean'] = eds_mean
            data['te_5_ec_mean'] = ec_mean
        elif k.startswith('inliers'):
            data['inliers_eds_mean'] = eds_mean
            data['inliers_ec_mean'] = ec_mean

    # Add columns for individual sequences
    all_sequences = []
    for sequence_name in SUPPORTED_SEQUENCES_FEATURE_TRACKING['eds']:
        all_sequences.append((sequence_name, EvalDatasetType.EDS))
    for sequence_name in SUPPORTED_SEQUENCES_FEATURE_TRACKING['ec']:
        all_sequences.append((sequence_name, EvalDatasetType.EC))
        
    for i, eval_sequence in enumerate(all_sequences):
        sequence_name = eval_sequence[0]
        for k in ['age_5', 'te_5', 'inliers']:
            column_name = f"{k}_{sequence_name}"
            header.append(column_name)
            data[column_name] = tables[f"{k}_mu"].rows[i][1]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(data)


# Data Classes for Inference
class EvalDatasetType(Enum):
    EC = 0
    EDS = 1

plt.rcParams["font.family"] = "serif"

def parse_args():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: An object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Parse paths for results and output directories."
    )
    parser.add_argument(
        'method',
        type=Path,
        help="Path to the output directory."
    )

    parser.add_argument(
        '--results_dir',
        type=Path,
        default=Path("output/inference"),
        help="Path to the results directory."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    error_threshold_range = np.arange(1, 32, 1)
    methods = [args.method]

    table_keys = [
        "age_5_mu",
        "age_5_std",
        "te_5_mu",
        "te_5_std",
        "age_mu",
        "age_std",
        "inliers_mu",
        "inliers_std",
        "expected_age",
    ]
    tables = {}
    for k in table_keys:
        tables[k] = PrettyTable()
        tables[k].title = k
        tables[k].field_names = ["Sequence Name"] + methods

    # Create a list of sequences with their dataset types
    eval_sequences = []
    for sequence_name in SUPPORTED_SEQUENCES_FEATURE_TRACKING['eds']:
        eval_sequences.append((sequence_name, EvalDatasetType.EDS))
    for sequence_name in SUPPORTED_SEQUENCES_FEATURE_TRACKING['ec']:
        eval_sequences.append((sequence_name, EvalDatasetType.EC))

    for eval_sequence in eval_sequences:
        sequence_name = eval_sequence[0]
        sequence_type = eval_sequence[1]

        gt_folder_name = 'eds' if sequence_type == EvalDatasetType.EDS else 'ec'
        track_data_gt = read_txt_results(
            Path('config/misc') / gt_folder_name / 'gt_tracks' / f"{sequence_name}.gt.txt"
        )

        rows = {}
        for k in tables.keys():
            rows[k] = [sequence_name]

        for method in methods:
            inlier_ratio_arr, fa_rel_nz_arr = [], []

            track_data_pred = read_txt_results(
                str(args.results_dir / f"{method}" / f"{sequence_name}.txt")
            )

            if track_data_pred[0, 1] != track_data_gt[0, 1]:
                raise ValueError  # TODO: double check if this case occurs
                track_data_pred[:, 1] += -track_data_pred[0, 1] + track_data_gt[0, 1]

            for thresh in error_threshold_range:
                fa_rel, _ = compute_tracking_errors(
                    track_data_pred,
                    track_data_gt,
                    error_threshold=thresh,
                    asynchronous=False,
                )

                inlier_ratio = np.sum(fa_rel > 0) / len(fa_rel)
                if inlier_ratio > 0:
                    fa_rel_nz = fa_rel[np.nonzero(fa_rel)[0]]
                else:
                    fa_rel_nz = [0]
                inlier_ratio_arr.append(inlier_ratio)
                fa_rel_nz_arr.append(np.mean(fa_rel_nz))

            mean_inlier_ratio, std_inlier_ratio = np.mean(inlier_ratio_arr), np.std(
                inlier_ratio_arr
            )
            mean_fa_rel_nz, std_fa_rel_nz = np.mean(fa_rel_nz_arr), np.std(fa_rel_nz_arr)
            expected_age = np.mean(np.array(inlier_ratio_arr) * np.array(fa_rel_nz_arr))

            rows["age_mu"].append(mean_fa_rel_nz)
            rows["age_std"].append(std_fa_rel_nz)
            rows["inliers_mu"].append(mean_inlier_ratio)
            rows["inliers_std"].append(std_inlier_ratio)
            rows["expected_age"].append(expected_age)

            fa_rel, te = compute_tracking_errors(
                track_data_pred, track_data_gt, error_threshold=5, asynchronous=False
            )
            inlier_ratio = np.sum(fa_rel > 0) / len(fa_rel)
            if inlier_ratio > 0:
                fa_rel_nz = fa_rel[np.nonzero(fa_rel)[0]]
            else:
                fa_rel_nz = [0]
                te = [0]

            mean_fa_rel_nz, std_fa_rel_nz = np.mean(fa_rel_nz), np.std(fa_rel_nz)
            mean_te, std_te = np.mean(te), np.std(te)
            rows["age_5_mu"].append(mean_fa_rel_nz)
            rows["age_5_std"].append(std_fa_rel_nz)
            rows["te_5_mu"].append(mean_te)
            rows["te_5_std"].append(std_te)

        # Load results
        for k in tables.keys():
            tables[k].add_row(rows[k])

    with open((args.results_dir / f"{method}" / f"benchmarking_results.csv"), "w") as f:
        for k in tables.keys():
            f.write(f"{k}\n")
            f.write(tables[k].get_csv_string())
            
            # Calculate and write mean values for EDS and EC
            eds_mean = calculate_mean_by_dataset_type(tables[k], EvalDatasetType.EDS)
            ec_mean = calculate_mean_by_dataset_type(tables[k], EvalDatasetType.EC)
            
            f.write(f"EDS Mean,{eds_mean}\n")
            f.write(f"EC Mean,{ec_mean}\n\n")

            print(tables[k].get_string())
            print(f"EDS Mean: {eds_mean}")
            print(f"EC Mean: {ec_mean}\n")

    summary_csv_path = args.results_dir / f"{method}" / f"summary_results.csv"
    create_summary_csv(tables, summary_csv_path)

    print(f"Summary results written to {summary_csv_path}")