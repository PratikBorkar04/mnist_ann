import os
import sys
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model,save_model
from src.utils.save_plot import save_plot
from src.utils.callbacks import get_callbacks
import argparse

def training(config_path):
    config = read_config(config_path)

    validation_datasize = config['params']["validation_datasize"]
    (X_train,y_train),(X_valid,y_valid),(X_test,y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = config['params']["loss_function"]
    OPTIMIZER = config['params']["optimizer"]
    METRICES = config['params']["metrics"]
    NUM_CLASSES = config['params']["num_classes"]
    EPOCHS = config['params']["epochs"]


    model = create_model(LOSS_FUNCTION,OPTIMIZER,METRICES,NUM_CLASSES)
    VALIDATION_SET = (X_valid,y_valid)

    callback_list = get_callbacks(config,X_train)

    history = model.fit(X_train,y_train,epochs = EPOCHS,validation_data=VALIDATION_SET,callbacks = callback_list)

    artifacts_dir = config['artifacts']["artifacts_dir"]
    model_name = config['artifacts']["model_name"]
    model_dir = config['artifacts']["model_dir"]

    model_dir_path  = os.path.join(artifacts_dir,model_dir)
    os.makedirs(model_dir_path,exist_ok=True)
    save_model(model,model_name,model_dir)

    plots_dir = config["artifacts"]["plots_dir"]
    plots_name = config["artifacts"]["plots_name"]

    save_plot(pd.DataFrame(history.history),plots_dir,plots_name)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)