# src/news/gen_news.py


# Use example: python -m src.news.1_gen_news --model gemini-1.5-flash


import os
from pathlib import Path
import yaml



import json
import pandas as pd
import numpy as np
import time
import getpass
import asyncio
import logging
import uuid
import argparse
from tenacity import retry, stop_after_attempt, wait_exponential



from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOctoAI
from langchain_anthropic import ChatAnthropic


from langchain_core.prompts import ChatPromptTemplate

base_dir = Path(__file__).resolve().parent.parent.parent
#base_dir = Path.cwd()
#base_dir = base_dir.parent.parent

print(base_dir)


from .. import models


model_list = [
    'claude-3-sonnet-20240229', 
    'gemini-1.5-flash', 
    'meta-llama-3-70b-instruct', 
    'qwen1.5-32b-chat', 
    'mixtral-8x22b-instruct'
]





def get_news_prompts(whichdata):
    prompt_directory = base_dir / "src/prompts"

    if whichdata == '8each':
        with open(prompt_directory / 'news_both_8each.json') as infile:
            prompts = json.load(infile)
    else:
        raise ValueError("Not a valid dataset to get prompts!")
    print("# prompts found: ", str(len(prompts)))
    return prompts




whichdata = '8each'
prompts = get_news_prompts(whichdata)




news_data =  base_dir / "data/news/fontes-media.csv"
df = pd.read_csv(news_data)
df.columns = ['name', 'vertical', 'ideology']


output_folder= base_dir / "output/news/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)




# NB: models [2:5] are run with octoml. openai models are run by batch.


def run_model(selected_model):
    print(f"Running with model: {selected_model}")
    if selected_model in ('claude-3-sonnet-20240229', 'gemini-1.5-flash'):
        sleeptime = 1.1 # less than 1/s
    else:
        sleeptime = 0.7

    # (already in models.py):
    # if selected_model == 'gemini-1.5-flash':
    #     if "GOOGLE_API_KEY" not in os.environ:
    #         os.environ["GOOGLE_API_KEY"] = getpass.getpass("provide google key")
    # elif selected_model =='claude-3-sonnet-20240229':
    #     if "ANTHROPIC_API_KEY" not in os.environ:
    #         os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("provide anthropic key")
    # else:
    #     if "OCTOAI_API_TOKEN" not in os.environ:
    #         os.environ["OCTOAI_API_TOKEN"] = getpass.getpass("provide ggl key")


    allerrs = []


    print("\n------------")
    print(selected_model)
    model, model_save_name, temp, _ = get_model(selected_model)

    rows = []

    for i, prmpt in prompts.items():
        prompt = ChatPromptTemplate.from_template(prmpt)
        chain = prompt | model

        for j, row in df.iterrows():
            currname = row['name']
            print(selected_model[:3], str(i), str(j), "/", str(len(df)), ":", currname)
            try:
                response = chain.invoke({"name": currname})
            except:
                print("FAILED!")
                time.sleep(3)
                try:
                    response = chain.invoke({"name": currname})
                except:
                    print("failed 2nd time! ---------")
                    allerrs.append([selected_model, currname, i, j, prmpt ])
                    errs.append([selected_model, currname, i, j])
                    continue
            text = response.content
            print(f"Response: {text[:70]}")
            rows.append([row['name'], text, selected_model, prmpt, row['vertical'], row['ideology'], response, time.time()])

            time.sleep(sleeptime)


    dfo = pd.DataFrame(rows, columns = ['name','text', 'model', 'prompt', 'vertical','ideology','resp','timestamp'])
    print("Len DF: ", str(len(dfo)))
    dfo.to_csv(output_folder + "news_" + selected_model + "_" + whichdata + ".csv", index=False)

    print(allerrs)
    errdf = pd.DataFrame(allerrs)
    errdf.to_csv(output_folder + "news_errs_" + selected_model + "_" + whichdata + ".csv", index=False)
    print("done.")





def main():
    parser = argparse.ArgumentParser(description="Run the script with a given llm.")
    parser.add_argument(
        '--model',
        type=str,
        choices=model_list,
        required=True,
        help="Choose which model to run."
    )
    args = parser.parse_args()
    run_model(args.model)





if __name__ == '__main__':
    main()




