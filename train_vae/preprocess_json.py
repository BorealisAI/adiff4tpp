# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import pandas as pd

# Function to load JSON data
def load_json_data(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)
    return json_data

# Function to convert JSON data to pandas DataFrame
def json_to_dataframe(json_data):
    data = []
    # Iterate through each sequence in the JSON data
    for sequence in json_data:
        time_since_last_event = sequence['time_since_last_event']
        type_event = sequence['type_event']
        # Zip time_since_last_event and type_event into pairs and append them to the data list
        data.extend(list(zip(time_since_last_event, type_event)))
    
    # Create a DataFrame with columns 'time_since_last_event' and 'type_event'
    df = pd.DataFrame(data, columns=['time_since_last_event', 'type_event'])
    return df

if __name__ == "__main__":
    # Path to your JSON file
    for data in ['taxi','amazon','stackoverflow','retweet','taobao']:
        for i in ['train','test']:
            filepath = data+'/'+i+'.json'

            # Load JSON data
            json_data = load_json_data(filepath)

            # Convert JSON data to DataFrame
            df = json_to_dataframe(json_data)
            df.to_csv(data+'_'+i+'.csv', index=False)
