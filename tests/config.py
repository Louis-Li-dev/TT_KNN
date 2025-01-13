import argparse
from typing import Union
# Create the parser
parser = argparse.ArgumentParser(description="Configuration for the game")

# Add the health parameter
parser.add_argument('--user_range', type=int, default=3_000, help='range of uids')
parser.add_argument('--data_folder', type=str, default="../cityB-dataset", help='where the data is stored')
parser.add_argument('--result_folder', type=str, default="../result", help='place to store the result csv files')



args = parser.parse_args()


USERS_RANGE = args.user_range
TAR_FOLDER = args.data_folder
RESULT_DIR = args.result_folder