from sklearn.neighbors import NearestNeighbors
import pandas as pd

def temporal_knn_fit_predict(
    df: pd.DataFrame,
    start_t: int = 0,
    end_t: int = 48,
    stride: int = 1,
    starting_coor: tuple = None,
    thresh: int = 4
) -> pd.DataFrame:
    """
    Predict the coordinates of a point over time using K-nearest neighbors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'x' and 'y' coordinates, with a 'd' column indicating time intervals.
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
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing predicted coordinates for each time step.
    """
    df = df.copy()

    counter = df[['x', 'y']].value_counts()
    top = counter[counter > thresh].index.tolist()
    df = df[df.apply(lambda row: (row['x'], row['y']) in top, axis=1)]
    

    model_intervals = [(i, i + stride) for i in range(start_t, end_t, stride)]
    
    model_dict, data_dict = {}, {}
    
    for interval in model_intervals:
        data = df[df.d.isin(range(interval[0], interval[1]))][['x', 'y']].drop_duplicates().reset_index(drop=True)
        
        if not data.empty:
            data_dict[interval] = data.values.tolist()
            model = NearestNeighbors(n_neighbors=2, n_jobs=-1, algorithm='ball_tree', leaf_size=40)
            model.fit(data_dict[interval])
            model_dict[interval] = model

    curr_coord = tuple(df[['x', 'y']].iloc[-1]) if starting_coor is None else starting_coor
    
    # Predict coordinates for each time step
    result = []
    for t in range(start_t + 1, end_t + 1):
        t %= 48 # when the timestamp is at 48, the prediction range lies in timestamp 0
        interval = next((i for i in model_intervals if t in range(i[0], i[1])), None)
   
        if interval and interval in model_dict:
            try:
                _, indices = model_dict[interval].kneighbors([curr_coord])
                curr_coord = data_dict[interval][indices[0][-1]]
            except Exception:
                pass
        
        result.append({'t': t, 'x': curr_coord[0], 'y': curr_coord[1]})

    return pd.DataFrame(result)
