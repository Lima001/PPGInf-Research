import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def get_metrics(y_true, y_pred):
    """
    Computes classification metrics for a single task.

    Returns a dictionary containing:
        "micro_acc": Overall accuracy (equivalent to `accuracy_score`).
        "macro_acc": Macro-averaged accuracy (unweighted mean of per-class accuracy). This is useful for imbalanced datasets.
        "macro_f1": Macro-averaged F1-score (unweighted mean of per-class F1-scores).

    Handles empty lists by returning zeros and suppresses warnings for classes with no predictions in F1-score calculation.
    """

    if not y_true:
        return {"micro_acc": 0.0, "macro_acc": 0.0, "macro_f1": 0.0}

    # Micro-average accuracy: calculates accuracy globally by counting the total number of correct predictions.
    micro_acc = accuracy_score(y_true, y_pred)
    
    # Macro-average accuracy: calculates accuracy for each class individually and then averages them (treating all classes equally).
    macro_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Macro-average F1-score: calculates F1 for each class and finds their unweighted mean. Does not take label imbalance into account.
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {"micro_acc": micro_acc, "macro_acc": macro_acc, "macro_f1": macro_f1}


import numpy as np
# Assuming 'task_metrics' function is defined elsewhere as in the previous example
# from your_metrics_file import task_metrics 

def compute_metrics(y_true_dict, y_pred_dict, task_classes, task_weights=None):
    """
    Computes and performs a weighted average of metrics across multiple tasks.

    Args:
        y_true_dict: A dictionary mapping task names to lists of true labels.
        y_pred_dict: A dictionary mapping task names to lists of predicted labels.
        task_classes: A dictionary mapping task names to the number of classes.
        task_weights: An optional dictionary mapping task names to weights for averaging. If None, equal weighting is used.

    Returns:
        A dictionary with 'micro_acc', 'macro_acc', and 'macro_f1' keys,
        representing the weighted average of these metrics across all tasks.
    """
    
    per_task_metrics = {}
    weights_list = []

    # Calculate metrics for each task individually
    for task_name in task_classes:
        y_true = y_true_dict.get(task_name, [])
        y_pred = y_pred_dict.get(task_name, [])
        per_task_metrics[task_name] = get_metrics(y_true, y_pred)
        
        # Assign weight for averaging
        if task_weights and task_name in task_weights:
            weights_list.append(task_weights[task_name])
        else:
            weights_list.append(1.0)

    # Normalize weights to sum to 1
    weights = np.array(weights_list)
    weights /= weights.sum()

    # Calculate the weighted average of each metric across all tasks
    avg_metrics = {
        "micro_acc": 0.0,
        "macro_acc": 0.0,
        "macro_f1": 0.0,
    }
    
    task_names = list(task_classes.keys())
    for metric_name in avg_metrics:
        metric_values = np.array([per_task_metrics[task][metric_name] for task in task_names])
        avg_metrics[metric_name] = float(np.sum(metric_values * weights))

    return avg_metrics, per_task_metrics