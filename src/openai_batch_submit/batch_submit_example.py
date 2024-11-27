import os
import json


#---------------
# submit batch
#---------------

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Provide your OPENAI Key")




client = OpenAI()


# example to submit the news data from gen_news_openai.py:

# (Need to have already created the jsonl file)
batch_input_file = client.files.create(
  file=open(output_folder + "news_gpt35_batch.jsonl", "rb"),
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
      "description": "35 news"
    }
)


#--- list ------
client.batches.list(limit=10)


# latest batch
bt = client.batches.list(limit=1)
bt.data[0]


outfile= bt.data[0].output_file_id



#--- if using batch id ...
#res = client.batches.retrieve(batchid) # batchid has form: "batch-..."
#outputfile = res.output_file_id
#print(outputfile)

content = client.files.content(outfile)

content.write_to_file(output_folder + "news_gpt35_batch_responses.jsonl")


# convert to df, combine with metadata, save to csv.

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


#----
# additional processing to match the output of the other files:

dfo = load_jsonl_to_df(output_folder + "news_gpt35_batch_responses.jsonl")

dfo['response.status_code'].value_counts() # check if errors

resps = dfo['response.body.choices']

output = [x[0]['message']['content'] for x in resps]


# ideally want output to match the rest - [row['name'], text, whichmodel, prmpt, row['vertical'], row['ideology'], response, time.time()])


dfm['text'] = output
dfm['response'] = [x[0] for x in resps]

dfm['whichmodel'] = 'gpt-3.5-turbo'

dfm['timestamp'] = dfo['response.body.created']

dfm2 = dfm[['name', 'text','whichmodel','fullprompt','vertical','ideology', 'response',  'timestamp']]

whichmodel = 'gpt-3.5-turbo'
whichdata = '8each'
dfm2.to_csv(output_folder + "news_" + whichmodel + "_" + whichdata + ".csv", index=False)



