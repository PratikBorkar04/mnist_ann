import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.utils.common import read_config
from src.utils.data_mgmt import get_data
import argparse

def training(config_path):
    config = read_config(config_path)
    validation_datasize = config['params']["validation_datasize"]
    (X_train,y_train),(X_valid,y_valid),(X_test,y_test) = get_data(validation_datasize)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)