from sklearn.neighbors import NearestNeighbors
import pandas as pd
def temporal_knn_fit_predict(
    df: pd.DataFrame,
    start_t: int = 0,
    end_t: int = 48,
    stride: int = 1,
    starting_coor: tuple = None,
    thresh: int = 4,
    n_neighbor: int = 2,
    val_cols: list['str'] = ['x', 'y'],
    t_col: str = 't'
) -> pd.DataFrame:
    """
    Predict the coordinates of a point over time using K-nearest neighbors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing coordinate columns and a 'd' column indicating time intervals.
    start_t : int, optional, default=0
        Minimum time (t) in the dataframe.
    end_t : int, optional, default=48
        Maximum time (t) in the dataframe.
    stride : int, optional, default=1
        Step size for the training time window. For each time step t, training data will be from (t, t+stride).
    starting_coor : tuple, optional, default=None
        Starting coordinates for prediction. If None, the last observation in the dataframe will be used.
    thresh : int, optional, default=4
        A threshold for the outliers to be removed. If a place has been visited less than thresh, it is filtered out.
    n_neighbors : int, optional, default=2
        The number of neighbors the model will find,
    val_cols : list['str'], optional, default=['x', 'y']
        A list of columns for the model to refer to. For example, in the mobility prediction task, x and y serve as coordinates in the dataframe
    t_col : str, optional, default = 't'
        The timestamp column.
    Returns
    -------
    pd.DataFrame
        DataFrame containing predicted coordinates for each time step.
    """
    if not isinstance(val_cols, list) or len(val_cols) < 1:
        raise ValueError("`val_cols` must be a list with at least one column name.")
    
    df = df.copy()
    
    # Ensure all specified columns exist in the DataFrame
    for col in val_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' specified in `val_cols` does not exist in the DataFrame.")
    
    # Count occurrences of the unique combinations in `val_cols`
    counter = df[val_cols].value_counts()
    top = counter[counter > thresh].index.tolist()
    
    # Filter out rows that do not have top-occurring coordinate combinations
    df = df[df.apply(lambda row: tuple(row[col] for col in val_cols) in top, axis=1)]
    
    model_intervals = [(i, i + stride) for i in range(start_t, end_t, stride)]
    
    model_dict, data_dict = {}, {}
    
    for interval in model_intervals:
        # Extract data for the current interval using the specified columns
        data = df[df['t'].isin(range(interval[0], interval[1]))][val_cols].drop_duplicates().reset_index(drop=True)
        
        if not data.empty:
            data_dict[interval] = data.values.tolist()
            model = NearestNeighbors(n_neighbors=n_neighbor, n_jobs=-1, algorithm='kd_tree', leaf_size=40)
            model.fit(data_dict[interval])
            model_dict[interval] = model
    
    # Determine the starting coordinates
    if starting_coor is None:
        if df.empty:
            raise ValueError("DataFrame is empty after filtering. Cannot determine starting coordinates.")
        curr_coord = tuple(df[val_cols].iloc[-1])
    else:
        if len(starting_coor) != len(val_cols):
            raise ValueError(f"`starting_coor` must have {len(val_cols)} elements.")
        curr_coord = starting_coor
    
    # Predict coordinates for each time step
    result = []
    for t in range(start_t + 1, end_t + 1):
        t_mod = t % 48  # when the timestamp is at 48, the prediction range lies in timestamp 0
        interval = next((i for i in model_intervals if t_mod in range(i[0], i[1])), None)
   
        if interval and interval in model_dict:
            try:
                _, indices = model_dict[interval].kneighbors([curr_coord])
                # Select the nearest neighbor (excluding itself if present)
                nearest_index = indices[0][-1]
                curr_coord = data_dict[interval][nearest_index]
            except Exception as e:
                pass
    
        # Construct the result dictionary with dynamic column names
        prediction = {'t': t_mod}
        for idx, col in enumerate(val_cols):
            prediction[col] = curr_coord[idx]
        result.append(prediction)
    
    return pd.DataFrame(result)
