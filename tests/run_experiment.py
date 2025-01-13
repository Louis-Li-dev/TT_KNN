
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import sys
parent_dir = os.path.join(os.getcwd(), '..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from config import USERS_RANGE, TAR_FOLDER, RESULT_DIR

def main():
    print(f"[VAR] uid range {USERS_RANGE}; data folder: {TAR_FOLDER}; saving folder: {RESULT_DIR}")
    '''Custom packages'''
    from ttknn import ttknn
    from ttknn.light_utility import Utility, write_unseen_rows
    import geobleu
    print("[PACK] Imported Packages ✅")
    city = TAR_FOLDER.split('/')[-1]
    print(f"[DATA] Dealing with data from {city}")
    tar_path_list = []
    for file in os.listdir(TAR_FOLDER):
        curr_dir = os.path.join(os.getcwd(), TAR_FOLDER)
        if file.endswith('.csv'):
            print(f"[DATA] File {file} detected.")
            tar_path_list.append(os.path.join(curr_dir, file))
    df = pd.read_csv(tar_path_list[0])
    
    all_uids = df.uid.unique()[:USERS_RANGE]
    processing_uids = all_uids
    train_df, test_df = Utility.test_train_split(df, 60)


    results = []
    '''Detect Remaining UIDs for Processing'''
    if os.path.exists(f"{RESULT_DIR}/{city}_file.csv"):
        curr_df = pd.read_csv(f"{RESULT_DIR}/{city}_file.csv")
        curr_uids = curr_df['uid'].unique()
        processing_uids  = set(all_uids).difference(curr_uids)
    print(f"[DATA] Length of remaining processing uids: {len(processing_uids)}")
    print("[PROG] Predicting Started")
    for i, uid in enumerate(tqdm(processing_uids)):
        sample = train_df[train_df.uid == uid]
        test_sample = test_df[test_df.uid == uid]

        submission_df = test_sample[['d', 't']]
        predicted = ttknn.temporal_knn_fit_predict(df=sample)
        s1 = Utility.to_eval_format(submission_df.merge(predicted, on='t', how='left'))
        s2 = Utility.to_eval_format(test_sample)
        
        s1_converted = [(int(d), int(t), int(x), int(y)) for (d, t, x, y) in s1]
        s2_converted = [(int(d), int(t), int(x), int(y)) for (d, t, x, y) in s2]

        results.append({
            "uid": uid,
            "s1": str(s1_converted),  # Now these are pure Python ints
            "s2": str(s2_converted)
        })
        
        # Save intermediate results every n steps

        intermediate_df = pd.DataFrame(results)
        write_unseen_rows(intermediate_df, f"{RESULT_DIR}/{city}_file.csv")
    print("[PROG] Predicting Finished 👍")
    data = pd.read_csv(RESULT_DIR)
    tqdm.pandas()
    full_dtw = []
    print("[PROG] Evaluating Initiated.")
    def get_dtw(row):
        s1 = eval(row['s1'])
        s2 = eval(row['s2'])
        if isinstance(s1, np.ndarray): s1 = s1.tolist()
        if isinstance(s2, np.ndarray): s2 = s2.tolist()
        try:
            dtw = geobleu.calc_dtw(s1, s2, processes=2)
        except Exception:
            dtw = None 
        full_dtw.append({'uid': row['uid'], 'dtw': dtw})
        write_unseen_rows(pd.DataFrame(full_dtw), f"{RESULT_DIR}/{city}_score.csv")

    data[data.uid.isin(processing_uids)].progress_apply(get_dtw, axis=1)
    
    print("[PROG] Evaluating completed.")
if __name__ == "__main__":
    main()