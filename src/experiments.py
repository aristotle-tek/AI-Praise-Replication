
import json
import pandas as pd
import os
import itertools
from pathlib import Path


base_path = Path(__file__).resolve().parent.parent


code_path = base_path / "src"






def create_prompt_dict(batch, field1, field2):
    if field2:
        return [{field1: row[field1], field2: row[field2]} for idx, row in batch.iterrows()]
    else:
        return [{field1: row[field1]} for idx, row in batch.iterrows()]



def get_news_prompts(whichdata):
    prompt_directory = "/prompts/"

    if whichdata == '8each':
        with open(prompt_directory + 'news_both_8each.json') as infile:
            prompts = json.load(infile)
    else:
        raise ValueError("Not a valid dataset to get prompts!")
    print("# prompts found: ", str(len(prompts)))
    return prompts



def get_moraldir(whichdata):
    # moral direction paper - Schramowski et al Nature Machine Intel 2022
    # combined with pre- and post- phrases.
    #batchfiles_folder = "/data/batchfiles_35/"
    batchfiles_folder = base_path / "data" / "values" / "batchfiles"
    df = pd.read_csv(batchfiles_folder + "batch_data_01_df.csv")

    if whichdata == "sents":
        prompts = list(set(df.sents))
    elif whichdata == "sentsrev":
        prompts = list(set(df.sentsRev))
    else:
        raise ValueError("Not a valid dataset!")
    return prompts, df




def international_leaders_data(whichdata):
    # Source: augmented+ https://en.wikipedia.org/wiki/List_of_current_heads_of_state_and_government
    intl_leaders_file = "data/politics/intl_pol_leaders.xlsx"
    df = pd.read_excel(filepath)
    return df




def get_world_leaders_prompts(whichdata):
    prompt_directory = "src/prompts/"
    with open(prompt_directory + 'intl_politician_prompts.json') as infile:
            prompts = json.load(infile)
    else:
        raise ValueError("Not a valid dataset to get prompts!")
    print("# prompts found: ", str(len(prompts)))
    return prompts




