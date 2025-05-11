import pandas as pd
import numpy as np

def calculate_accuracy(predicted_df, actual_df):
    """
    Compares predicted fire spread with actual data.
    Args:
        predicted_df (pd.DataFrame): Predictions with 'node_id', 'ignition_time', 'prediction_step_count'.
        actual_df (pd.DataFrame): Actual fire data with 'node_id', 'ignition_time'.
    Returns:
        dict: Accuracy metrics.
    """
    if predicted_df.empty:
        return {"error": "No predictions to evaluate."}
    if actual_df.empty:
        return {"error": "No actual data to evaluate against."}

    # Ensure ignition_time is datetime
    predicted_df['ignition_time'] = pd.to_datetime(predicted_df['ignition_time'])
    actual_df['ignition_time'] = pd.to_datetime(actual_df['ignition_time'])
    
    # Spatial Accuracy (Set-based comparison of newly ignited nodes)
    # Exclude initial fires (prediction_step_count == 0) from predicted_df for fair comparison
    predicted_newly_ignited_nodes = set(predicted_df[predicted_df['prediction_step_count'] > 0]['node_id'])
    
    # Actual data might include initial fires. For a fair comparison of *newly* spread fires,
    # one might filter actual_df based on a time window or exclude known initial fire nodes if possible.
    # Here, we consider all nodes in actual_df for recall calculation regarding newly predicted ones.
    actual_ignited_nodes = set(actual_df['node_id'])
    
    # Nodes that were part of the initial set (not predicted as new)
    initial_fire_nodes_in_prediction = set(predicted_df[predicted_df['prediction_step_count'] == 0]['node_id'])

    # True Positives: Newly predicted nodes that actually ignited (and were not initial fires in prediction set)
    tp_nodes = predicted_newly_ignited_nodes.intersection(actual_ignited_nodes)
    # False Positives: Newly predicted nodes that did not actually ignite
    fp_nodes = predicted_newly_ignited_nodes.difference(actual_ignited_nodes)
    
    # False Negatives: Nodes that actually ignited but were not among the *newly* predicted nodes.
    # (This excludes nodes that were correctly identified as initial fires by the prediction setup)
    actual_newly_ignited_for_fn = actual_ignited_nodes.difference(initial_fire_nodes_in_prediction)
    fn_nodes = actual_newly_ignited_for_fn.difference(predicted_newly_ignited_nodes)

    precision = len(tp_nodes) / len(predicted_newly_ignited_nodes) if predicted_newly_ignited_nodes else 0
    recall = len(tp_nodes) / len(actual_newly_ignited_for_fn) if actual_newly_ignited_for_fn else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # print(f"Spatial Accuracy (newly ignited nodes):")
    # print(f"  Predicted new ignitions: {len(predicted_newly_ignited_nodes)}")
    # print(f"  Actual new ignitions considered for recall: {len(actual_newly_ignited_for_fn)}")
    # print(f"  True Positives (correctly predicted new): {len(tp_nodes)}")
    # print(f"  False Positives (predicted new, not actual): {len(fp_nodes)}")
    # print(f"  False Negatives (actual new, not predicted as new): {len(fn_nodes)}")
    # print(f"  Precision: {precision:.2f}")
    # print(f"  Recall: {recall:.2f}")
    # print(f"  F1 Score: {f1:.2f}")

    mae_hours, rmse_hours = None, None
    if tp_nodes: # Temporal accuracy only for True Positives
        merged_df = pd.merge(
            predicted_df[predicted_df['node_id'].isin(tp_nodes)][['node_id', 'ignition_time']].rename(columns={'ignition_time': 'predicted_time'}),
            actual_df[actual_df['node_id'].isin(tp_nodes)][['node_id', 'ignition_time']].rename(columns={'ignition_time': 'actual_time'}),
            on='node_id', how='inner'
        )
        if not merged_df.empty:
            merged_df['time_diff_hours'] = (merged_df['predicted_time'] - merged_df['actual_time']).dt.total_seconds() / 3600.0
            mae_hours = merged_df['time_diff_hours'].abs().mean()
            rmse_hours = np.sqrt((merged_df['time_diff_hours']**2).mean())
            # print(f"Temporal Accuracy (MAE for TP nodes): {mae_hours:.2f} hours")
            # print(f"Temporal Accuracy (RMSE for TP nodes): {rmse_hours:.2f} hours")
        else:
            # print("No common TP nodes found for temporal accuracy after merge.")
            pass # MAE/RMSE will remain None
    # else:
        # print("No true positive nodes for temporal accuracy calculation.")

    return {
        'precision': precision, 'recall': recall, 'f1_score': f1,
        'tp_count': len(tp_nodes), 'fp_count': len(fp_nodes), 'fn_count': len(fn_nodes),
        'mae_hours': mae_hours, 'rmse_hours': rmse_hours
    } 