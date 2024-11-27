
# NB: Gemini 1.5-flash, Gemini 1.0 - free - max 15/m = 4s wait



import json
import pandas as pd
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


code_path = base_path / "src"


output_folder = base_path / "data" / "values"


os.chdir(code_path)

from models import * # get_model, get_octo_model(), etc.

from experiments import get_moraldir # * #


#if "OPENAI_API_KEY" not in os.environ:
#    os.environ["OPENAI_API_KEY"] = getpass.getpass("Provide your OPENAI Key")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key") # ...4

if "OCTOAI_API_TOKEN" not in os.environ:
    os.environ["OCTOAI_API_TOKEN"] = getpass.getpass("Provide your OCTOAI api token")

# temp 0 to 1, defaults to 1.0
if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Provide your ANTHROPIC API Key")






def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def setup_logging(log_directory, run_id):
    log_file = f"{log_directory}log_{run_id}.log"
    os.makedirs(log_directory, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )



def create_prompt_dict(batch, field1, field2):
    if field2:
        return [{field1: row[field1], field2: row[field2]} for idx, row in batch.iterrows()]
    else:
        return [{field1: row[field1]} for idx, row in batch.iterrows()]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=20))
async def call_api_with_retry(chain, prompt_dicts):
    return await chain.abatch(prompt_dicts)



model_list = ['gpt-3.5-turbo', 'claude-3-sonnet-20240229', 'gemini-1.5-flash', \
'qwen1.5-32b-chat','meta-llama-3-70b-instruct',"mixtral-8x22b-instruct"]


errs = []


for whichmodel in model_list:

    if whichmodel[:6] == 'gemini':
        sleeptime = 4.3
    else:
        sleeptime=1.3


    for whichdata in ['sents','sentsrev']:
        prompts, df = get_moraldir(whichdata)

        rows = []
        
        print("\n------------")
        print(whichmodel)
        model, model_save_name, temp, callrate = get_model(whichmodel)


        i = 0
        print("running item-wise...")
        for prmpt in prompts:
            print(str(i), prmpt)

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", prmpt),
            ]
            try:
                response = model.invoke(messages)
            except:
                    print("FAILED!")
                    time.sleep(3)
                    try:
                        response = model.invoke(messages)
                    except:
                        print("failed 2nd time! ---------")
                        errs.append([whichmodel, prmpt, i])
                        continue

            text = response.content
            print(f"Response: {text[:70]}")
            rows.append([i, text, whichmodel, prmpt, response, time.time()])

            time.sleep(sleeptime)
            i +=1


        dfo = pd.DataFrame(rows, columns = ['idx','text', 'model', 'prompt','resp','timestamp'])
        print("Len DF: ", str(len(dfo)))
        filename = "values_" + whichmodel + "_"+ whichdata + "01.csv"
        outfile = output_folder / filename
        dfo.to_csv(outfile, index=False)


        print(errs)
        print("done.")
