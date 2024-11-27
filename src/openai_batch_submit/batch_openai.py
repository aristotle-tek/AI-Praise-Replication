
from openai import OpenAI

import pandas as pd


import itertools
import json
import numpy as np
import re


base_json_structure = {
    "custom_id": "",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-3.5-turbo-0125",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ""}
        ],
        "max_tokens": 1000
    }
}


def handle_invalid_json(df):
    # Check for NaN values (or batch may fail)
    invalid_rows = df[
        df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    ]
    
    if not invalid_rows.empty:
        print("Found invalid JSON values in the following rows:")
        print(invalid_rows)

    # Replace invalid values with None (or any other placeholder value)
    df.replace([np.nan, np.inf, -np.inf], None, inplace=True)

    return df



def create_jsonl(prompts):
    jsonl_content = []
    for i, prompt in enumerate(prompts):
        json_object = base_json_structure.copy()
        json_object["custom_id"] = f"request-{i+1}"
        json_object["body"]["messages"][1]["content"] = prompt
        jsonl_content.append(json.dumps(json_object))
    return jsonl_content



def is_valid_json(json_string):
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False


# jsonL files
def is_valid_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if not is_valid_json(line.strip()):
                    return False
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def get_last_part_of_path(path):
    pattern = r'[^/]+$'
    match = re.search(pattern, path)
    if match:
        return match.group()
    return path[-48:]



def load_jsonl(file_path):
    """
    Load a JSONL file to dict
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
    return data


def load_jsonl_to_df(file_path):
    """
    Load a JSONL file to df
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

    return pd.json_normalize(data)


