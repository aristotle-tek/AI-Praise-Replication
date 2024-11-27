
import re
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


import itertools

from pathlib import Path


base_path = Path(__file__).resolve().parent.parent

code_path = base_path / "src"

os.chdir(code_path)

from batch_openai import *




system_message = "You are a careful, thoughtful text analysis and text-coding assistant."

base_json_structure = {
    "custom_id": "",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-3.5-turbo-0125",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": ""}
        ],
        "max_tokens": 1000
    }
}




output_folder=base_path / "data" / "world_leaders" / "output" 


model_list = ['gpt-3.5-turbo', 'claude-3-sonnet-20240229', 'gemini-1.5-flash', \
'qwen1.5-32b-chat','meta-llama-3-70b-instruct',"mixtral-8x22b-instruct"]



#------------------------------------
# create a set of batches
#------------------------------------


batchfile_list = []

for whichmodel in model_list[1:]:
    print("Model: ", whichmodel)
    currfile = output_folder + "wldrs_" + whichmodel + "_"+ whichdata+ ".csv"
    df = pd.read_csv(currfile)
    outfile = re.sub('.csv','_batch.jsonl', currfile)
    batchfile_list.append(outfile)
    print(len(df))
    

    textprompts = []

    for i, row in df.iterrows():

        response = row['text']
        prompt_w_text = evalprompt.format(response)
        textprompts.append([prompt_w_text, row['name'], row['state'], row['cat'],row['text'], row['timestamp']])

    pdf = pd.DataFrame(textprompts, columns=['tosubmit','name','state','cat', 'text','timestamp'])
    df_file = re.sub('.jsonl','_df.csv', outfile)

    pdf.to_csv(df_file, index=False)
    allprompts = list(pdf['tosubmit'])
    jsonl_content = create_jsonl(allprompts)


    with open(outfile, "w") as jsonl_file:
        for line in jsonl_content:
            jsonl_file.write(line + "\n")

    if is_valid_jsonl(outfile):
        print("JSONL file created successfully.")
    else:
        print("ERROR! invalid json? -- ", outfile)
    time.sleep(3.3)



# GPT-3.5 - different because batch....

whichmodel = model_list[0]
print("Model: ", whichmodel)
currfile = output_folder + "wldrs_" + whichmodel + "_"+ whichdata+ ".csv"
#if whichmodel == 'gpt-3.5-turbo': # (exception because ran as batch)
filepath = '/output/returns/intl_gpt-3.5-turbo_5each01.jsonl'
df = load_jsonl_to_df(filepath)
outfile = re.sub('.csv','_batch.jsonl', filepath)

batchfile_list.append(outfile)
print(len(df))


textprompts = []

for i, row in df.iterrows():

    response = row['response.body.choices'][0]['message']['content'] # row['text']
    prompt_w_text = evalprompt.format(response)
    textprompts.append([prompt_w_text,  row['custom_id'] ])

pdf = pd.DataFrame(textprompts, columns=['tosubmit','custom_id'])
df_file = re.sub('.jsonl','_df.csv', outfile)

pdf.to_csv(df_file, index=False)
allprompts = list(pdf['tosubmit'])
jsonl_content = create_jsonl(allprompts)


with open(outfile, "w") as jsonl_file:
    for line in jsonl_content:
        jsonl_file.write(line + "\n")

if is_valid_jsonl(outfile):
    print("JSONL file created successfully.")
else:
    print("ERROR! invalid json? -- ", outfile)




#------------------------------------
# submit a set of batches.
#------------------------------------





from openai import OpenAI


client = OpenAI()

# upload

bfilelist = []

for bfile in batchfile_list[:]:
    filen = get_last_part_of_path(bfile)
    print(filen)

    batch_input_file = client.files.create(
      file=open(bfile, "rb"),
      purpose="batch"
    )

    # create the batch
    batch_input_file_id = batch_input_file.id
    print("inputfileid: ", batch_input_file_id)

    bfilelist.append(batch_input_file_id)
    # submit
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
          "description":  filen # "values_ eval job"
        }
    )
    print('completed.')
    time.sleep(5.5)




# wldrs_claude-3-sonnet-20240229_5each_batch.jsonl
# inputfileid:  file-HG8NrH27qoHoQUxc9lQEBCwP
# completed.
# wldrs_gemini-1.5-flash_5each_batch.jsonl
# inputfileid:  file-7yAr9Lgf1APfWtMwwf9XyotJ
# completed.
# wldrs_qwen1.5-32b-chat_5each_batch.jsonl
# inputfileid:  file-4ORsrrHC1TgO8OGPGJJ046hy
# completed.
# wldrs_meta-llama-3-70b-instruct_5each_batch.jsonl
# inputfileid:  file-Whbiv1ZPtwBmVjmjvwMma3Xo
# completed.
# wldrs_mixtral-8x22b-instruct_5each_batch.jsonl
# inputfileid:  file-id71nHMmuFVsTUlvjPnJNV5B
# completed.
# intl_gpt-3.5-turbo_5each01.jsonl
# inputfileid:  file-r3RL3VZYgMka2NeIAzSWBzJr
# completed.





for bfile in bfilelist[:]:
    print(bfile)
    try:
        res = client.batches.retrieve(bfile)
        outputfile = res.output_file_id
        content = client.files.content(outfile)
        descr = res.metadata['description']
        descr = re.sub(r'[\s]+', '-', str(descr) )
        content.write_to_file(output_folder + "news_gpt35_eval_responses" + str(descr)  + ".jsonl")
    except:
        print('failed for: ', bfile)




# latest 6 batchs
bt = client.batches.list(limit=6)


#--- print only ---
info = []
for bth in bt.data:
    descriptor = bth.metadata['description']
    outfile_id = bth.output_file_id
    print(descriptor)
    print(bth.status)
    print(outfile_id)
    print('---')



#---- save --- 
info = []
for bth in bt.data:
    descriptor = bth.metadata['description']
    modelname = re.sub(r'wldrs_', '', descriptor)
    modelname = re.sub(r'_5each_batch.jsonl','', modelname)
    print(descriptor)
    print(bth.status)
    try:
        outfile_id = bth.output_file_id
        print(outfile_id)
        content = client.files.content(outfile_id)
        print('got file, writing...')
        content.write_to_file(output_folder + "intlpol_eval_"+ modelname + ".jsonl")
    except:
        print("not able to save.")
    print('---')
    time.sleep(3)




#-------------------------------------------------
# convert data, correct errors where possible
#-------------------------------------------------

def load_jsonl(file_path):
    """
    Load a JSON Lines file and return a list of dictionaries.

    :param file_path: str, path to the .jsonl file
    :return: list of dictionaries
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
    Load a JSON Lines file into a pandas DataFrame.

    :param file_path: str, path to the .jsonl file
    :return: pandas DataFrame
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


def extract_content(row):
    """
    Extract the 'content' field from the nested JSON structure in the DataFrame row.

    :param row: pandas Series, a row of the DataFrame
    :return: str, extracted content
    """
    try:
        return row[0]['message']['content']
    except (IndexError, KeyError, TypeError):
        return None


def code_string_output(s: str) -> int:
    # Check for p -1, 0, +1 (or 1)
    # has_negative_one = bool(re.search(r'(?<!\d)-1(?!\d)', s))
    # has_zero = bool(re.search(r'(?<!\d)0(?!\d)', s))
    # has_positive_one = bool(re.search(r'(?<!\d)\+?1(?!\d)', s))
    
    pattern_negative_one = r'-\s*1'
    pattern_zero = r'\b0\b'
    pattern_positive_one = r'\+?\s*1'
    
    # Search for the patterns in the string
    has_negative_one = re.search(pattern_negative_one, s) is not None
    has_zero = re.search(pattern_zero, s) is not None
    has_positive_one = re.search(pattern_positive_one, s) is not None

    # verify not conflicting values where possible
        # note - 
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


#  eval files - 



eval_folder=base_path / "data" / "world_leaders" / "output" 

eval_files = ["intlpol_eval_claude-3-sonnet-20240229.jsonl","intlpol_eval_gemini-1.5-flash.jsonl",
        "intlpol_eval_gpt-3.5-turbo_5each01.jsonl",
        "intlpol_eval_meta-llama-3-70b-instruct.jsonl",
        "intlpol_eval_mixtral-8x22b-instruct.jsonl",
        "intlpol_eval_qwen1.5-32b-chat.jsonl"]



pattern = re.compile(r'intpol_eval_(\w+)\.jsonl')




for filename in eval_files:
    file = eval_folder + filename
    #print(file)
    jfc = load_jsonl_to_df(file)
    print(jfc.shape)
    whichmodel = re.sub('intpol_eval_', '', filename)
    whichmodel = re.sub('.jsonl', '', whichmodel)
    print(whichmodel)

    jfc['content'] = jfc['response.body.choices'].apply(lambda x: extract_content(x))
    jfc['code'] = jfc['content'].apply(lambda x: code_string_output(x))
    print("to correct: ", str(len(jfc[jfc.code==999])))
    jfc = correct_errors(jfc)
    jfc.to_csv(eval_folder + "intpol_evalcoded_" + whichmodel + ".csv", index=False)


