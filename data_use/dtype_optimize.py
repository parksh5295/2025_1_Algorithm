import pandas as pd
import numpy as np


# Estimate dtype while memory-efficiently sampling a CSV
def infer_dtypes_safely(csv_path, max_rows=1000, chunk_size=100):
    chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size)
    inferred_dtypes = None
    row_count = 0

    for chunk in chunk_iter:
        if inferred_dtypes is None:
            column_names = list(chunk.columns)
            inferred_dtypes = {col: set() for col in column_names}

        for _, row in chunk.iterrows():
            for col in column_names:
                val = row[col]

                '''
                # If the column should be forced to str, add "str" to its dtype set
                if force_str_columns is not None and col in force_str_columns:
                    inferred_dtypes[col].add("str")
                    continue
                '''

                if pd.isnull(val):
                    continue
                if isinstance(val, int):
                    inferred_dtypes[col].add("int")
                elif isinstance(val, float):
                    inferred_dtypes[col].add("float")
                elif isinstance(val, str):
                    inferred_dtypes[col].add("str")
                else:
                    inferred_dtypes[col].add("other")
            row_count += 1
            if row_count >= max_rows:
                break
        if row_count >= max_rows:
            break

    print("[DEBUG] inferred_dtypes type:", type(inferred_dtypes))
    print("[DEBUG] inferred_dtypes sample:", str(inferred_dtypes)[:500])

    # dtype estimation rules
    dtype_map = {}
    for col, types in inferred_dtypes.items():
        if types <= {"int"}:
            dtype_map[col] = 'int32'
        elif types <= {"int", "float"}:
            dtype_map[col] = 'float32'
        elif types <= {"str"}:
            dtype_map[col] = 'object'
        else:
            print(f"[WARN] Unknown type set {types} for column '{col}', using object")
            dtype_map[col] = 'object'

    return dtype_map


# Efficiently load a full CSV into a DataFrame after auto-estimating dtype
def load_csv_safely(file_type, csv_path, max_rows_for_inference=1000):
    print("[INFO] Estimating: Sampling for dtype inference...")
    dtype_map = infer_dtypes_safely(file_type, csv_path, max_rows=max_rows_for_inference)
    print("dtype map: ", dtype_map)
    print("[INFO] Dtype inference complete. Loading full CSV...")
    df = pd.read_csv(csv_path, dtype=dtype_map, low_memory=False)
    print("[INFO] Finished loading the DataFrame:", df.shape)
    return df


def optimize_loaded_df(df, exclude_cols=None):
    """
    Optimizes memory usage for a DataFrame or GeoDataFrame that is already loaded into memory.

    Args:
        df (pd.DataFrame or gpd.GeoDataFrame): The DataFrame to optimize.
        exclude_cols (list, optional): List of column names to exclude from optimization.
                                      For GeoDataFrames, the geometry column is automatically excluded.

    Returns:
        pd.DataFrame or gpd.GeoDataFrame: the optimized DataFrame.
    """
    optimized_df = df.copy()
    original_mem = optimized_df.memory_usage(deep=True).sum() / 1024**2

    cols_to_exclude = set(exclude_cols) if exclude_cols else set()

    # if GeoDataFrame, automatically exclude geometry column
    # check if geopandas is installed and if df is a GeoDataFrame
    try:
        import geopandas as gpd
        if isinstance(optimized_df, gpd.GeoDataFrame):
            cols_to_exclude.add(optimized_df.geometry.name)
            print(f"[INFO] Automatically excluding geometry column: {optimized_df.geometry.name}")
    except ImportError:
        pass # skip GeoDataFrame check if geopandas is not installed

    print(f"[INFO] Columns to exclude from optimization: {cols_to_exclude}")

    for col in optimized_df.columns:
        if col in cols_to_exclude:
            continue

        col_type = optimized_df[col].dtype

        try: # try to convert types that may cause errors
            if pd.api.types.is_integer_dtype(col_type):
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                if pd.isna(c_min) or pd.isna(c_max): # Handle columns with NaNs
                    continue # Skip optimization if min/max can't be determined due to all NaNs
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                    optimized_df[col] = optimized_df[col].astype(np.int64)
            elif pd.api.types.is_float_dtype(col_type):
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                if pd.isna(c_min) or pd.isna(c_max): # Handle columns with NaNs
                    continue
                # Consider float32 (float16 has limited precision)
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    # Check if conversion to float32 maintains precision reasonably
                    if np.allclose(optimized_df[col].dropna().astype(np.float32), optimized_df[col].dropna()):
                         optimized_df[col] = optimized_df[col].astype(np.float32)
                    # else: keep float64 if precision loss is significant
                # else: keep float64
            elif col_type == 'object':
                # Convert to category if it saves memory (heuristic: unique values < 50%)
                num_unique_values = optimized_df[col].nunique()
                num_total_values = len(optimized_df[col])
                if num_unique_values / num_total_values < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')

        except Exception as e:
             print(f"[WARN] Could not optimize column '{col}' of type {col_type}. Error: {e}")


    optimized_mem = optimized_df.memory_usage(deep=True).sum() / 1024**2
    print("\n--- Dataframe Memory Optimization ---")
    print(f"Original memory usage: {original_mem:.2f} MB")
    print(f"Optimized memory usage: {optimized_mem:.2f} MB")
    mem_reduction = (original_mem - optimized_mem) / original_mem * 100 if original_mem > 0 else 0
    print(f"Memory reduced by: {mem_reduction:.2f}%")
    print("------------------------------------")

    return optimized_df
