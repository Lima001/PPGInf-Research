# -*- coding: utf-8 -*-
"""
Stratified K-Fold Split Generator with Identifier Exclusivity

This script partitions a dataset, described by a JSON annotations file, into
K folds for cross-validation. Its primary goal is to create high-quality,
balanced splits suitable for robust machine learning model training and evaluation.

The script implements two key principles:

1.  **Stratification across Multiple Attributes:**
    It ensures that the distribution of user-specified attributes (e.g., 'color',
    'make') is as consistent as possible across all K folds. If the full dataset
    contains 10% red items, each fold will also contain approximately 10% red
    items. This prevents distributional bias and ensures that each split is a
    representative sample of the overall dataset.

2.  **Identifier Exclusivity:**
    The script guarantees that all data samples sharing the same identifier
    (e.g., multiple images of the same unique object) are kept together in the
    *same fold*. This is critical for preventing data leakage, where a model
    could be unfairly tested on data that is nearly identical to what it saw
    during training, leading to inflated performance metrics. The JSON key for this
    identifier can be specified via the `--id-key` argument.

Algorithm Overview:
- The script first loads all data and groups samples by their identifier.
- It calculates the overall distribution of every attribute class for the given
  attributes.
- To achieve a balanced partition, it prioritizes the placement of ID groups
  that contain the rarest attribute classes, as these are the most difficult
  to distribute evenly.
- It iteratively assigns each ID group to the fold that will experience the
  least increase in distributional imbalance.
- Finally, it outputs K '.txt' files, each containing the filenames for one
  fold, and prints a detailed summary of the final attribute distributions.

Example Usage:
    python stratify.py annotations.json --output ./folds --folds 5 --attributes color make model type --id_key unique_vehicle_id
"""
import argparse
import json
import os
import random
from collections import defaultdict
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Create stratified K-fold splits with identifier exclusivity")
    parser.add_argument("json_file", help="Path to JSON annotations")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--folds", type=int, required=True, help="Number of cross-validation folds (K)")
    parser.add_argument("--attributes", nargs='+', required=True, help="List of attributes to stratify on")
    parser.add_argument("--id_key", type=str, required=True, help="The JSON key for the identifier that groups samples"
    )
    args = parser.parse_args()

    # Load and process data
    with open(args.json_file) as f:
        data = json.load(f)

    K = args.folds
    ATTRIBUTES = args.attributes
    ID_KEY = args.id_key

    # Group data by ID
    id_groups = defaultdict(list)
    attribute_counts = {attr: defaultdict(int) for attr in ATTRIBUTES}

    for obj in data:
        if ID_KEY not in obj:
            print(f"Warning: ID key '{ID_KEY}' not found in object with filename '{obj.get('filename', 'N/A')}'. Skipping.")
            continue
        id_val = str(obj[ID_KEY]).strip()
        id_groups[id_val].append(obj)
        for attr in ATTRIBUTES:
            if attr not in obj:
                print(f"Warning: Attribute '{attr}' not found in object with filename '{obj.get('filename', 'N/A')}'. Skipping.")
                continue
            val = str(obj[attr]).strip() if isinstance(obj[attr], str) else obj[attr]
            attribute_counts[attr][val] += 1

    ids = list(id_groups.keys())
    random.shuffle(ids)  # Initial shuffle for randomness

    # Prepare folds
    folds = [
        {
            'ids': set(),
            'counts': {attr: defaultdict(int) for attr in ATTRIBUTES}
        } for _ in range(K)
    ]

    # Sort IDs by how rare their attributes are
    def rarity_score(id_val):
        score = 0
        for obj in id_groups[id_val]:
            for attr in ATTRIBUTES:
                # Ensure attribute exists in the object before calculating score
                if attr in obj and obj[attr] in attribute_counts[attr] and attribute_counts[attr][obj[attr]] > 0:
                    score += 1 / (attribute_counts[attr][obj[attr]] ** 2)
        return score

    id_priority = sorted(ids, key=rarity_score, reverse=True)

    # Distribute IDs into folds with stratification
    for id_val in id_priority:
        group = id_groups[id_val]
        attr_contributions = {attr: defaultdict(int) for attr in ATTRIBUTES}
        for obj in group:
            for attr in ATTRIBUTES:
                if attr not in obj:
                    continue
                val = str(obj[attr]).strip() if isinstance(obj[attr], str) else obj[attr]
                attr_contributions[attr][val] += 1

        # Find the fold where adding this ID group minimizes attribute imbalance
        best_fold = None
        min_error = float('inf')

        for i, fold in enumerate(folds):
            current_error = 0
            for attr in ATTRIBUTES:
                for cls, cnt in attr_contributions[attr].items():
                    current = fold['counts'][attr][cls]
                    ideal = attribute_counts[attr][cls] / K
                    current_error += abs((current + cnt) - ideal) - abs(current - ideal)
            if current_error < min_error:
                min_error = current_error
                best_fold = i

        # Assign ID to the best fold
        folds[best_fold]['ids'].add(id_val)
        for attr in ATTRIBUTES:
            for cls, cnt in attr_contributions[attr].items():
                folds[best_fold]['counts'][attr][cls] += cnt

    # Output folds
    os.makedirs(args.output, exist_ok=True)
    for i, fold in enumerate(folds):
        filename = os.path.join(args.output, f"fold_{i}.txt")
        with open(filename, "w") as f:
            # Sort IDs for deterministic output
            for id_val in sorted(list(fold['ids'])):
                for obj in id_groups[id_val]:
                    f.write(f"{obj['filename']}\n")

        # Print distribution summary
        total_in_fold = sum(len(id_groups[id_val]) for id_val in fold['ids'])
        
        print(f"\n--- Fold {i} ({total_in_fold} samples) ---")
        if total_in_fold == 0:
            continue

        for attr in ATTRIBUTES:
            print(f"  [ {attr.capitalize()} Distribution ]")
            
            # Determine padding for alignment based on the longest class name
            max_cls_len = 0
            if fold['counts'][attr]:
                max_cls_len = max(len(str(cls)) for cls in fold['counts'][attr].keys())

            # Sort by class name for consistent reporting
            for cls, cnt in sorted(fold['counts'][attr].items()):
                ideal = attribute_counts[attr][cls] / K
                percentage = (cnt / total_in_fold) * 100
                
                # Format the line for aligned, readable output
                cls_str = f"{cls}:".ljust(max_cls_len + 2)
                count_str = f"{cnt}".rjust(4)
                percent_str = f"({percentage:.1f}%)".ljust(8)
                ideal_str = f"Ideal: {ideal:.1f},"
                diff_str = f"Diff: {cnt - ideal:+.1f}"

                print(f"    - {cls_str} {count_str} {percent_str} | {ideal_str} {diff_str}")

if __name__ == "__main__":
    main()