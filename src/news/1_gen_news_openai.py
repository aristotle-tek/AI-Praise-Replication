# similar to gen_news.py, but generates a batch (cheaper and easier) for openai gpt-3.5-turbo


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
from tenacity import retry, stop_after_attempt, wait_exponential



from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOctoAI
from langchain_anthropic import ChatAnthropic


from openai import OpenAI
    

import itertools


import json


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
    # Check for NaN values (or batch can fail)
    invalid_rows = df[
        df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    ]
    
    if not invalid_rows.empty:
        print("Found invalid JSON values in the following rows:")
        print(invalid_rows)

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




base_dir = Path(__file__).resolve().parent.parent.parent

print(base_dir)

from .. import models



def get_news_prompts(whichdata):
    prompt_directory = base_dir / "src" / "prompts"

    if whichdata == '8each':
        with open(prompt_directory / 'news_both_8each.json') as infile:
            prompts = json.load(infile)
    else:
        raise ValueError("Not a valid dataset to get prompts!")
    print("# prompts found: ", str(len(prompts)))
    return prompts




whichdata = '8each'
prompts = get_news_prompts(whichdata)




news_data =  base_dir / "data" / "news" / "fontes-media.csv"
df = pd.read_csv(news_data)
df.columns = ['name', 'vertical', 'ideology']


output_folder= base_dir / "output"/ "news"



if not os.path.exists(output_folder):
    os.makedirs(output_folder)





def main():
    rows = []


    for i, prmpt in prompts.items():
        #prompt = ChatPromptTemplate.from_template(prmpt)
        for j, row in df.iterrows():
            currname = row['name']
            fullprompt = prmpt.format(name=currname)
            print(fullprompt)
            rows.append([i, j, currname, prmpt, fullprompt, row['vertical'], row['ideology']]) # to ensure assign to metadata...


    dfm = pd.DataFrame(rows)
    dfm.columns = ['prompt_idx', 'name_idx', 'name', 'promptbase', 'fullprompt', 'vertical', 'ideology']


    promptslist = list(dfm.fullprompt)


    jsonl_content = create_jsonl(promptslist)



    with open(output_folder + "news_gpt35_batch.jsonl", "w") as jsonl_file:
        for line in jsonl_content:
            jsonl_file.write(line + "\n")

    print("JSONL file created successfully.")


# Then submit the batch.
# see src/batch_submit_example.py for submitting batch code.


