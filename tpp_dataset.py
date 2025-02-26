# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch.utils.data import Dataset
import pandas as pd
from easy_tpp.utils import load_pickle
import os

data_config_dict = {
    "taxi": {
        "data_format": "pkl",
        "train_dir": "train_vae/taxi/train.pkl",
        "valid_dir": "train_vae/taxi/dev.pkl",
        "test_dir": "train_vae/taxi/test.pkl",
        "data_spec": {
            "num_event_types": 10,
            "max_len": 38,
            "pad_token_id": 0,
            "padding_side": "right",
            "truncation_side": "right"
        }
    },
    "taobao": {
        "data_format": "pkl",
        "train_dir": "train_vae/taobao/train.pkl",
        "valid_dir": "train_vae/taobao/dev.pkl",
        "test_dir": "train_vae/taobao/test.pkl",
        "data_spec": {
            "num_event_types": 20,
            "max_len": 64,
            "pad_token_id": 0,
            "padding_side": "right",
            "truncation_side": "right"
        }
    },
    "amazon": {
        "data_format": "pkl",
        "train_dir": "train_vae/amazon/train.pkl",
        "valid_dir": "train_vae/amazon/dev.pkl",
        "test_dir": "train_vae/amazon/test.pkl",
        "data_spec": {
            "num_event_types": 16,
            "max_len": 94,
            "pad_token_id": 0,
            "padding_side": "right",
            "truncation_side": "right"
        }
    },
    "stackoverflow": {
        "data_format": "pkl",
        "train_dir": "train_vae/stackoverflow/train.pkl",
        "valid_dir": "train_vae/stackoverflow/dev.pkl",
        "test_dir": "train_vae/stackoverflow/test.pkl",
        "data_spec": {
            "num_event_types": 22,
            "max_len": 101,
            "pad_token_id": 0,
            "padding_side": "right",
            "truncation_side": "right"
        }
    },
    "retweet": {
        "data_format": "pkl",
        "train_dir": "train_vae/retweet/train.pkl",
        "valid_dir": "train_vae/retweet/dev.pkl",
        "test_dir": "train_vae/retweet/test.pkl",
        "data_spec": {
            "num_event_types": 3,
            "max_len": 97,
            "pad_token_id": 0,
            "padding_side": "right",
            "truncation_side": "right"
        }
    }
}

def generate_csv_datasets():
    for dataname in data_config_dict.keys():
        config_dict = data_config_dict[dataname]
        train_data = load_pickle(config_dict["train_dir"])["train"]
        valid_data = load_pickle(config_dict["valid_dir"])["dev"]
        test_data = load_pickle(config_dict["test_dir"])["test"]

        # Convert JSON data to DataFrame
        df = json_to_dataframe(train_data)
        df.to_csv("train_vae/"+dataname+'_train.csv', index=False)
        df = json_to_dataframe(valid_data)
        df.to_csv("train_vae/"+dataname+'_valid.csv', index=False)
        df = json_to_dataframe(test_data)
        df.to_csv("train_vae/"+dataname+'_test.csv', index=False)

# Function to convert JSON data to pandas DataFrame
def json_to_dataframe(json_data):
    data = []
    # Iterate through each sequence in the JSON data
    for i in json_data:
        for sequence in i:
            time_since_last_event = sequence['time_since_last_event']
            type_event = sequence['type_event']
            # Zip time_since_last_event and type_event into pairs and append them to the data list
            data.append((time_since_last_event,type_event))
    
    # Create a DataFrame with columns 'time_since_last_event' and 'type_event'
    df = pd.DataFrame(data, columns=['time_since_last_event', 'type_event'])
    return df

def generate_pt_datasets():
    for dataname in data_config_dict.keys():
        config_dict = data_config_dict[dataname]
        train_data = load_pickle(config_dict["train_dir"])["train"]
        valid_data = load_pickle(config_dict["valid_dir"])["dev"]
        test_data = load_pickle(config_dict["test_dir"])["test"]
        save_dir = "pt_dataset/"+dataname
        save_dataset_as_pt(train_data,config_dict["data_spec"],save_dir+"/train/")
        save_dataset_as_pt(valid_data,config_dict["data_spec"],save_dir+"/valid/")
        save_dataset_as_pt(test_data,config_dict["data_spec"],save_dir+"/test/")

def save_dataset_as_pt(dataset,data_spec,save_dir):
    max_len = data_spec["max_len"]
    num_events = len(dataset)
    X_num = torch.ones((num_events,max_len,1), dtype=torch.float) * data_spec["pad_token_id"]
    X_cat = torch.ones((num_events,max_len,1), dtype=torch.long) * data_spec["pad_token_id"]
    X_len = torch.ones((num_events), dtype=torch.long)
    os.makedirs(save_dir, exist_ok=True)
    for event in range(num_events):
        len_event = len(dataset[event])
        if data_spec["padding_side"] == "right":
            X_num[event,:len_event,0] = torch.tensor([dataset[event][idx_event]['time_since_last_event'] for idx_event in range(len_event)], dtype=torch.float)
            X_cat[event,:len_event,0] = torch.tensor([dataset[event][idx_event]['type_event'] for idx_event in range(len_event)], dtype=torch.long)
        else: # data_spec["padding_side"] == "left"
            X_num[event,-len_event:,0] = torch.tensor([dataset[event][idx_event]['time_since_last_event'] for idx_event in range(len_event)], dtype=torch.float)
            X_cat[event,-len_event:,0] = torch.tensor([dataset[event][idx_event]['type_event'] for idx_event in range(len_event)], dtype=torch.long)
        X_len[event] = len_event
    torch.save(X_num, save_dir+"num.pt")
    torch.save(X_cat, save_dir+"cat.pt")
    torch.save(X_len, save_dir+"len.pt")

def compute_max_len(config_dict):
    train_data = load_pickle(config_dict["train_dir"])["train"]
    valid_data = load_pickle(config_dict["valid_dir"])["dev"]
    test_data = load_pickle(config_dict["test_dir"])["test"]
    return max([max(len(dataset) for dataset in data) for data in [train_data, valid_data, test_data]])

class TPPDataset(Dataset):
    def __init__(self, pt_folder):
        super().__init__()
        self.X_num = torch.load(pt_folder+"/num.pt", weights_only=True)
        self.X_cat = torch.load(pt_folder+"/cat.pt", weights_only=True)
        self.X_len = torch.load(pt_folder+"/len.pt", weights_only=True)
        
    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]
        this_len = self.X_len[index]

        sample = (this_num, this_cat, this_len)

        return sample

    def __len__(self):
        return self.X_num.shape[0]

if __name__ == "__main__":
    generate_pt_datasets()
    generate_csv_datasets()
