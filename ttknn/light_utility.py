
import pandas as pd
from copy import deepcopy
import os
from sklearn.multioutput import MultiOutputRegressor # type: ignore
def write_unseen_rows(intermediate_df, file_path):
    """
    Write only unseen rows to the output file.
    
    Parameters:
    - intermediate_df: pandas.DataFrame, the new data to be written.
    - file_path: str, path to the output file.
    """
    if os.path.exists(file_path):
        # Load existing data
        existing_df = pd.read_csv(file_path)
        
        # Concatenate existing and new data, and drop duplicates
        combined_df = pd.concat([existing_df, intermediate_df], ignore_index=True).drop_duplicates()
        
        # Identify new rows by excluding existing rows
        unseen_rows = combined_df[len(existing_df):]
    else:
        # If the file doesn't exist, all rows are unseen
        unseen_rows = intermediate_df

    # Append only unseen rows to the file
    if not unseen_rows.empty:
        unseen_rows.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

class Utility:
    def __init__(self, **arg):
        '''
            ## Available functions
            1. __store_item(key, val)
            2. __get_item__(key)
            3. register_function(name, func)
            4. append_dummy(base_data, columns)
        '''
        self.library = arg
    def __store_item__(self, key, val):
        self.library[key] = val
    def __get_item__(self, key):
        if key in self.library:
            return self.library[key]
        else:
            print("Warning: no such key in the library")
            return None
    def register_function(self, name, func):
        if not callable(func):
            raise ValueError("Must pass in a function")
        else:
            self.library[name] = func
    def append_dummy(base_data: pd.DataFrame):
        '''
            ## base_data 
            # Concatenate the data frame with a blank row
            ## Example:  
            A pandas data frame is passed in as a parameter base_data 
            The passed-in data frame base_data will be added with a new row with blank values filled with -1 
        '''
        return pd.concat([base_data, pd.DataFrame(columns=base_data.columns, index=[0]).fillna(-1)], ignore_index=True)
    def to_dataframe(df: pd.DataFrame, date):
        '''
            format the values into the data frame
        '''
        data = deepcopy(df)
        data['d'] = date[0]
        data['t'] = date[1]
        data = data[df.columns]
        return data
    def incre_date(d_t):
        '''
            Increment the date by one: formula: \n
            t += 1\n
            d += t // 48\n
            t = t % 48\n
            48 is the number of time intervals in a day
        '''
        d, t = d_t
        t += 1
        d += t // 48
        t = t % 48
        return d, t
    def create_lag_features(
            df: pd.DataFrame, lags: int, forecasting_steps: int = -1,
            testing=False,
            x_columns = ['x', 'y', 'is_day', 'is_weekend', 'POI_category_0', 'POI_count_0', "POI_category_1", "POI_count_1", "POI_category_2", "POI_count_2"],
            y_columns = ['x', 'y'],
            subset = ['x']
        ) -> pd.DataFrame:
        '''
            <h2> Typical use: </h2>
            <li> Primarily consider params df, lags and forecasting_steps </li>
            <li> Pass in a data frame that needs transformation </li>
            <li> Lags and forecasting_steps are past and future features respectively </li>
        '''
        dropping_columns = [data for data in x_columns if data not in y_columns]
        lagged_features = {
            f"{col}_lag_{lag}": df[col].shift(lag)
            for lag in range(1, lags + 1)
            for col in x_columns
        }
        
        lagged_df = pd.DataFrame(lagged_features)
        
        if forecasting_steps != -1:
            forecast_features = {
                f"{col}_future_{future}": df[col].shift(-forecasting_steps)
                for future in range(1, forecasting_steps)
                for col in x_columns
            }
            forecast_features_df = pd.DataFrame(forecast_features)
            df = pd.concat([df.drop(dropping_columns, axis=1), lagged_df, forecast_features_df], axis=1)
        else:   
            df = pd.concat([df.drop(dropping_columns, axis=1), lagged_df], axis=1)
        return df.dropna(subset=subset) if testing else df.dropna()
    def xySplit(df: pd.DataFrame, forecasting_steps=-1):
        '''
            Return a x and a y.\n
            x contains rows with columns that are not columns x and y (coordinate)
            y contains rows with columns that are only columns x and y (coordinate)
        '''
        if forecasting_steps == -1:
            return df[[col for col in df.columns if col not in ['x', 'y']]].astype(int), df[['x', 'y']].astype(int)
        else:
            new_col = ['x', 'y'] + [col for col in df.columns if "future" in col]
            return df[[col for col in df.columns if col not in new_col]].astype(int), df[new_col].astype(int)

    def sliding_window_predict(model, window, data):
        '''
            Return the prediction of the given model and a window for the next prediction\n
            Pass the returned window into the param window and go in a loop to implement a sliding window 
        '''
        next_window = pd.concat([window.iloc[1:], data])
        next_added_data = Utility.create_lag_features(Utility.append_dummy(next_window), 30 * 48).drop(['uid', 'd', 't', 'x', 'y'], axis=1)
        return model.predict(next_added_data.astype(int)).astype(int)[0], next_window
    



    def build_training_dataset(data: list, lags=30 * 48, forecasting_steps=-1):
        '''
            <h2> Param Explained </h2>
            Param data is a data frame from a single user.\n
            Param target_uid is the uid you primarily focus on.\n
            Param city_index is the city you are analyzing.\n
            
            <h2> Return x and y for model training </h2>
            
        '''
        added_data = Utility.create_lag_features(data.copy(), lags, forecasting_steps)
        test_added_data = added_data[added_data['x'] != -1]
        x, y = Utility.xySplit(test_added_data.drop(['uid', 'd', 't'], axis=1), forecasting_steps)
        return x, y

    def test_train_split(data: pd.DataFrame, split_date):
        train_list = data[data.d < split_date]
        test_list = data[data.d >= split_date]
        return train_list, test_list
    def training(model, x, y, muti_predicting=True):
        '''
            Use a module that trains multiple models to predict different columns
        '''
        if muti_predicting:
            regressor = MultiOutputRegressor(model, n_jobs=-1)
        else:
            regressor = model
        regressor.fit(x.astype(int), y.astype(int))
        return regressor
    def feature_addition(d, t, x, y, POI_map, query_idx=0):
        '''
            <h2> Param Explained </h2>
            POI_map is the map about location info.\n
            query_idx is the label for the city accessed right now\n
            <h2> Return a data frame that contains complete info from the given data</h2>
        '''
        df = {}
        TOP_COL_NUMS = 3
        # Retrieve the relevant query map
        query_map = POI_map[query_idx]
        coord = (x, y)
        
        # Prepare lists for storing the new columns
        category_cols = [[] for _ in range(TOP_COL_NUMS)]
        POI_cols = [[] for _ in range(TOP_COL_NUMS)]
        
        if coord in query_map:
            categories = query_map[coord]['category']
            POI_counts = query_map[coord]['POI_count']
        else:
            categories = [-1] * TOP_COL_NUMS
            POI_counts = [-1] * TOP_COL_NUMS
        
        # Append values to corresponding lists
        for i in range(TOP_COL_NUMS):
            category_cols[i] = categories[i] if i < len(categories) else -1
            POI_cols[i] = POI_counts[i] if i < len(POI_counts) else -1
        df['d'] = d
        df['t'] = t
        df['x'] = x
        df['y'] = y
        df['is_day'] = t < 24
        df['is_weekend'] = d % 7 < 2
        # Assign the lists to the DataFrame
        for i in range(TOP_COL_NUMS):
            df[f'POI_category_{i}'] = category_cols[i]
            
            df[f'POI_count_{i}'] = POI_cols[i] if category_cols[i] != -1 else -1
        return pd.DataFrame(df, index=[0])
    
    def get_final_segment(df: pd.DataFrame, final_date):
        '''
            <h3> This function separates the segment of data that will be used to predict</h3>
            <h2> Param Explained </h2>
            data_list: a list of Data Frame. Each data frame contains data from a single city\n
            final_date: the amount of data used for prediction\n
            <h2> Return a list of data frames </h2>
        '''
        return df[df.d >= final_date]
    def to_eval_format(df: pd.DataFrame):
        focused_df = df[['d', 't', 'x', 'y']]
        res = []
        for _, row in focused_df.iterrows():
            res.append((row['d'], row['t'], row['x'], row['y']))
        return res
    def mask_prediction(prediction_df, submission_df, on='t'):
        return Utility.to_eval_format(submission_df.merge(prediction_df, on=on, how='left'))
    def update_global_dataframe(data, df, next_date):
        adding_data = Utility.to_dataframe(data, next_date)
        return pd.concat([df, adding_data])
    
    def calc_date(d_t):
        d, t = d_t
        t += 1
        d += t // 48
        t = t % 48
        return d, t

    def sequentialPredicting(model, window, data, lags=30*48):
        next_windows = pd.concat([window.iloc[1:], data])
        next_added_data = Utility.create_lag_features(Utility.append_dummy(next_windows), lags).drop(['uid', 'd', 't', 'x', 'y'], axis=1)

        return model.predict(next_added_data.astype(int)).astype(int)[0], next_windows
