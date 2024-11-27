
# data:
# https://en.wikipedia.org/wiki/List_of_current_heads_of_state_and_government


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

from pathlib import Path


base_path = Path(__file__).resolve().parent.parent

import itertools


code_path = base_path / "src"

os.chdir(code_path)


batchfiles_folder = base_path / "data" / "values" / "batchfiles"

from models import * # get_model, get_octo_model(), etc.

from experiments import get_world_leaders_prompts

# data/world_leaders/intl_pol_leaders_long.xlsx`
intl_leaders_file = base_path / "data" / "world_leaders" / "intl_pol_leaders_long.xlsx"

df = pd.read_excel(intl_leaders_file)



df.columns = ['name', 'state', 'headofstate', 'headofgov', 'category', 'notes']

whichdata = '5each'
prompts = get_world_leaders_prompts(whichdata)



output_folder=base_path / "data" / "world_leaders" / "output" 

model_list = ['claude-3-sonnet-20240229', 'gemini-1.5-flash', 'meta-llama-3-70b-instruct', \
'qwen1.5-32b-chat',"mixtral-8x22b-instruct"]





#------------------
# run octoml
#------------------


get_api_key("OCTOAI_API_KEY", "Provide your OCTOAI API key")


sleeptime = 0.7 # for individual calls.

allerrs = []


for whichmodel in model_list[2:]:
    print("\n------------")
    print(whichmodel)
    model, model_save_name, temp, callrate = get_model(whichmodel)

    rows = []
    errs = []

    for i, prmpt in prompts.items():
        prompt = ChatPromptTemplate.from_template(prmpt)
        chain = prompt | model

        for j, row in df.iterrows():
            currname = row['name']
            print(whichmodel[:3], str(i), str(j), "/", str(len(df)), ":", currname)
            try:
                response = chain.invoke({"name": currname})
            except:
                print("FAILED!")
                time.sleep(3)
                try:
                    response = chain.invoke({"name": currname})
                except:
                    print("failed 2nd time! ---------")
                    allerrs.append([whichmodel, currname, i, j, prmpt, row['state'] ])
                    errs.append([whichmodel, currname, i, j])
                    continue
            text = response.content
            print(f"Response: {text[:70]}")
            rows.append([row['name'], text, whichmodel, prmpt, row['state'], row['category'], response, time.time()])

            time.sleep(sleeptime)

    dfo = pd.DataFrame(rows, columns = ['name','text', 'model', 'prompt', 'state','cat','resp','timestamp'])
    print("Len DF: ", str(len(dfo)))
    dfo.to_csv(output_folder + "world_leaders/wldrs_" + whichmodel + "_" + whichdata + ".csv", index=False)


print(allerrs)
print("done.")





#----------------------
# google / anthropic
#----------------------

get_api_key("ANTHROPIC_API_KEY", "Provide your anthrop API key")


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("provide ggl key")



whichmodel = model_list[1] # google
whichmodel = model_list[0] # anthr

print("\n------------")
print(whichmodel)
model, model_save_name, temp, callrate = get_model(whichmodel)

sleeptime = 1.1 # google/ anthr



allerrs = []
rows = []


for i, prmpt in prompts.items():
    prompt = ChatPromptTemplate.from_template(prmpt)
    chain = prompt | model

    for j, row in df.iterrows():
        currname = row['name']
        print(whichmodel[:3], str(i), str(j), "/", str(len(df)), ":", currname)
        try:
            response = chain.invoke({"name": currname})
        except:
            print("FAILED!")
            time.sleep(3)
            try:
                response = chain.invoke({"name": currname})
            except:
                print("failed 2nd time! ---------")
                allerrs.append([whichmodel, currname, i, j, prmpt, row['state'] ])
                continue
        text = response.content
        print(f"Response: {text[:70]}")
        #  'nominate_dim1', 'nominate_dim2',
        rows.append([row['name'], text, whichmodel, prmpt, row['state'], row['category'], response, time.time()])

        time.sleep(sleeptime)

dfo = pd.DataFrame(rows, columns = ['name','text', 'model', 'prompt', 'state','cat','resp','timestamp'])
print("Len DF: ", str(len(dfo)))
dfo.to_csv(output_folder + "world_leaders/wldrs_" + whichmodel + "_" + whichdata + ".csv", index=False)


print(allerrs)
print("done.")


