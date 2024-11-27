
import re
import json
import pandas as pd
import numpy as np
import time
import getpass
import os
import asyncio
import logging
import uuid
import argparse


import itertools
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats


def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
    return data


def load_jsonl_to_df(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

    return pd.json_normalize(data)


def extract_content(row):
    try:
        return row[0]['message']['content']
    except (IndexError, KeyError, TypeError):
        return None


def code_string_output(s: str) -> int:
    # Check for p -1, 0, +1 (or 1)
    pattern_negative_one = r'-\s*1'
    pattern_zero = r'\b0\b'
    pattern_positive_one = r'\+?\s*1'
    
    # Search for the patterns in the string
    has_negative_one = re.search(pattern_negative_one, s) is not None
    has_zero = re.search(pattern_zero, s) is not None
    has_positive_one = re.search(pattern_positive_one, s) is not None

    # verify not conflicting values where possible
    if has_negative_one:
        if has_zero:
            print("multiple scores assigned...")
            return 999
        else:
            return -1
    elif has_zero:
        if has_positive_one:
            print("multiple scores assigned...")
            return 999
        else:
            return 0
    elif has_positive_one:
        return 1
    return 999


def correct_errors(df):
    # Hand validate errors
    #
    # Ensure there's a 'correctedcode' column in the df
    if 'correctedcode' not in df.columns:
        df['correctedcode'] = df['code']

    for idx, row in df.iterrows():
        if row['code'] == 999:
            print(f"Error found in row {idx}")
            print("Content: ", row['content'])
            while True:
                try:
                    corrected_code = int(input("Enter the corrected code (-1, 0, 1) or 999 if fail "))
                    if corrected_code in [-1, 0, 1, 999]:
                        df.at[idx, 'correctedcode'] = corrected_code
                        break
                    else:
                        print("Invalid code. Please enter -1, 0, or 1.")
                except ValueError:
                    print("Invalid input. Please enter an integer value.")
    return df
