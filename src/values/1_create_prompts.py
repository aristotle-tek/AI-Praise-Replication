
from openai import OpenAI

import json
import pandas as pd

import time


import getpass
import os


import requests
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import asyncio



#----------------------------
# from gen_morality_prompts / cartesian_
#---------------------------

import json
import pandas as pd
import time
import getpass
import os
import asyncio
import logging
import uuid
import argparse

from pathlib import Path


base_path = Path(__file__).resolve().parent.parent

import itertools


code_path = base_path / "src"

os.chdir(code_path)

from models import *

batchfiles_folder = base_path / "data" / "values" / "batchfiles"




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


def create_jsonl(prompts):
    jsonl_content = []
    for i, prompt in enumerate(prompts):
        json_object = base_json_structure.copy()
        json_object["custom_id"] = f"request-{i+1}"
        json_object["body"]["messages"][1]["content"] = prompt
        jsonl_content.append(json.dumps(json_object))
    return jsonl_content



def convert_to_sentences(actions):
    setup_phrase = [
        "",
        "I don't care what other people think, I have to ",
        "I've tried to be good all my life, but now I'm going to ",
        "I hate to think what my dad would think, but I've decided to ",
        "I'm finally going to "
    ]

    setup_phrase_r2 = [
        "",
        "my momma taught me to be good, but I'm gonna ",
        "I'm overwhelmed and I'm just gonna ",
    ]

    suffix = [
        ".",
        ", come what may.",
        ", even if I regret it.",
    ]

    suffix_r2 = [
        ".",
        ", and I ain't lookin back.",
        ", no regrets."
    ]

    # Get Cartesian product sentences
    combinations_r1 = list(itertools.product(setup_phrase, actions, suffix))  # register 1 (more formal)
    combinations_r2 = list(itertools.product(setup_phrase_r2, actions, suffix_r2))  # register 2 (less formal)

    actionslist = []
    sents1and2 = []
    sents_r1 = []
    sents_r2 = []

    for combination in combinations_r1:
        setup, action, end = combination
        sentence = ''.join(combination)
        sents_r1.append(sentence)
        sents1and2.append(sentence)
        actionslist.append(action)

    for combination in combinations_r2:
        setup, action, end = combination
        sentence = ''.join(combination)
        sents_r2.append(sentence)
        sents1and2.append(sentence)
        actionslist.append(action)

    print(len(combinations_r1))  # 1620
    print(len(combinations_r2))  # 972
    print(len(sents1and2)) 

    return sents1and2, actionslist


# --- data with added 'reversed' ("opposite") versions:

actions_file = base_path / "data" / "values" / "prompts_userglobal.xlsx"
df = pd.read_excel(actions_file)

df = df[df.exclude!=1]

actions = df['actionmod'] # Action - original

actionsrev = df['oppaction'] # Action - original



sents1and2, actionslist = convert_to_sentences(actions)

sents1and2rev, actionslistrev = convert_to_sentences(actionsrev)


#------
# For saving....

len(actionslistrev) == len(actionslist)
dfsave = pd.DataFrame(zip(sents1and2, actionslist))

dfsave.columns= ['sents','actions']
dfsave['sentsRev'] = sents1and2rev
dfsave['actionsRev'] = actionslistrev




dfsave.to_csv(batchfiles_folder + "batch_data_final_df.csv", index=False)


jsonl_content = create_jsonl(sents1and2)



with open(batchfiles_folder + "actionsrev_01.jsonl", "w") as jsonl_file:
    for line in jsonl_content:
        jsonl_file.write(line + "\n")

print("JSONL file created successfully.")







#----------
# submit batch

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Provide your OPENAI Key")




client = OpenAI()

batch_input_file = client.files.create(
  file=open(batchfiles_folder + "actions_01.jsonl", "rb"),
  purpose="batch"
)

# create the batch

batch_input_file_id = batch_input_file.id
print("batch id: ", batch_input_file_id)



client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "35 actions 01"
    }
)

#--- list ------
client.batches.list(limit=10)


batch_input_file2 = client.files.create(
  file=open(batchfiles_folder + "actionsrev_01.jsonl", "rb"),
  purpose="batch"
)

# create the batch

batch_input_file_id2 = batch_input_file2.id
print("batch id: ", batch_input_file_id2)


client.batches.create(
    input_file_id=batch_input_file_id2,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "35 actions REV 01"
    }
)


# check status, retrieve batch files

client = OpenAI()

client.batches.retrieve("batch_foo")
res = client.batches.retrieve("batch_foo")

outputfile = res.output_file_id
print(outputfile)

content = client.files.content("file-foo")

contentres = client.files.content(outputfile)

content.write_to_file(batchfiles_folder + "output_actions_01.jsonl")

contentres.write_to_file(batchfiles_folder + "output_actionsrev_01.jsonl")








