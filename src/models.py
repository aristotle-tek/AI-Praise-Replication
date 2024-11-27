# Warning - OctoML went out of business Oct 30, 2024. But other similar companies exist.

#--- rate limits ----

# octoml - 4/s
# openai - 9999
# google - 9999
# anthrop - ? 1/s ?



# conda activate langchain
# google-ai-generativelanguage 0.6.4                    pypi_0    pypi
# langchain                 0.1.14                   pypi_0    pypi
# langchain-anthropic       0.1.8                    pypi_0    pypi
# langchain-community       0.0.31                   pypi_0    pypi
# langchain-core            0.1.52                   pypi_0    pypi
# langchain-google-genai    1.0.5                    pypi_0    pypi
# langchain-google-vertexai 1.0.4                    pypi_0    pypi
# langchain-openai          0.0.8                    pypi_0    pypi
# langchain-text-splitters  0.0.1                    pypi_0    pypi
# langchainplus-sdk         0.0.17                   pypi_0    pypi
# langcodes                 3.3.0                    pypi_0    pypi
# langsmith                 0.1.40                   pypi_0    pypi



import json
import getpass
import os

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOctoAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import ChatPromptTemplate


# defaults - 
#     //  - openai 0.7
#     //  - Mixtral 0.7
#     //  - anthropic 1.0 (0-1)
#     //  - Google 1.0 (0-2) 
#     //  - Qwen - 1.0 (0-2)
#     //  - Llama3 70b - Reddit, LlamaIndex says 0.6; AWS says 0.5 (0-1); 
#     //         https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook/




def get_api_key(env_var, prompt):
    if env_var not in os.environ:
        os.environ[env_var] = getpass.getpass(prompt)
    return os.environ[env_var]


def get_model(whichmodel):
    # get_api_key("OPENAI_API_KEY", "Provide your OPENAI Key")
    # get_api_key("GOOGLE_API_KEY", "Provide your Google API Key")
    # get_api_key("OCTOAI_API_TOKEN", "Provide your OCTOAI API Token")
    # get_api_key("ANTHROPIC_API_KEY", "Provide your ANTHROPIC API Key")
    
    if whichmodel == 'gpt-3.5-turbo':
        model_save_name = 'gpt-35'
        temp = 0.7
        model = ChatOpenAI(temperature=temp, model=whichmodel)
        callrate = 999
    elif whichmodel == 'gpt-4o':
        model_save_name = 'gpt-4o'
        temp = 0.7
        model = ChatOpenAI(temperature=temp, model=whichmodel)
        callrate = 999
    elif whichmodel == 'gpt-4-turbo':
        model_save_name = 'gpt-4'
        temp = 0.7
        model = ChatOpenAI(temperature=temp, model=whichmodel)
        callrate = 999
    elif whichmodel == 'claude-3-sonnet-20240229':
        model_save_name = 'claude-3-sonnet'
        temp = 1.0
        model = ChatAnthropic(temperature=temp, model_name=whichmodel)
        callrate = 1
    elif whichmodel == 'claude-3-opus-20240229':
        model_save_name = 'claude-3-opus'
        temp = 1.0
        model = ChatAnthropic(temperature=temp, model_name=whichmodel)
        callrate = 1
    elif whichmodel == 'gemini-1.5-flash':
        model_save_name = 'gemini-1.5-flash'
        temp = 1.0
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temp)
        callrate = 1 # actually 999 but cheaper if low...
    elif whichmodel == 'qwen1.5-32b-chat':
        model_save_name = 'gwen15_32b'
        temp = 1.0
        model = ChatOctoAI(max_tokens=1500, model_name='qwen1.5-32b-chat', temperature=temp)
        callrate = 4 # per second
    elif whichmodel == 'meta-llama-3-70b-instruct':
        model_save_name = 'llama3-70b'
        temp = 0.6
        model = ChatOctoAI(max_tokens=1500, model_name=whichmodel, temperature=temp)
        callrate = 4 # per second
    elif whichmodel == "mixtral-8x22b-instruct":
        model_save_name = "mixtral-8x22b"
        temp = 0.7
        model = ChatOctoAI(max_tokens=1500, model_name=whichmodel, temperature=temp)
        callrate = 4 # per second
    elif whichmodel == 'llama3-7b':
        # ensure running...
        model_save_name = 'llama3-7b'
        temp = 0.6
        model = ChatOllama(max_tokens=1500, model_name=whichmodel, temperature=temp)
        callrate = 0 # wait to completion
    else:
        raise ValueError("Not a valid model!")

    return model, model_save_name, temp, callrate



def get_openai_chain(config):
    from langchain_openai import ChatOpenAI
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Provide your OPENAI Key")


    if config['model_name'] == 'gpt-3.5-turbo':
        model_save_name = 'gpt-35'
        model = ChatOpenAI(temperature=config["temperature"], model=config["model_name"]).bind(logprobs=True)
    elif config['model_name'] == 'gpt-4o': # current version: gpt-4o-2024-05-13
        model_save_name = 'gpt-4o'
        model = ChatOpenAI(temperature=config["temperature"], model=config["model_name"])
    elif config['model_name'] == 'gpt-4-turbo':
        model_save_name = 'gpt-4'
        model = ChatOpenAI(temperature=config["temperature"], model=config["model_name"])
    else:
        raise ValueError("Not a valid model!")

    prompt = ChatPromptTemplate.from_template(config['prompt'])
    chain = prompt | model

    return chain, model_save_name




def get_octo_chain(config):

    from langchain_community.chat_models import ChatOctoAI

    if "OCTOAI_API_TOKEN" not in os.environ:
        os.environ["OCTOAI_API_TOKEN"] = getpass.getpass("Provide your OCTOAI api token")

    if config['model_name'] == 'qwen1.5-32b-chat':
        model_save_name = 'gwen15_32b'
        model = ChatOctoAI(max_tokens=1500, model_name='qwen/qwen1.5-32b-chat', temperature=config["temperature"]) # specific modelname
    elif config['model_name'] == 'Llama-3-70b-Instruct':
        model_save_name = 'llama3-70b'
        model = ChatOctoAI(max_tokens=1500, model_name=config["model_name"], temperature=config["temperature"])
    elif config['model_name'] == "mixtral-8x22b-instruct":
        model_save_name = "mixtral-8x22b"
        model = ChatOctoAI(max_tokens=1500, model_name=config["model_name"], temperature=config["temperature"])
    else:
        raise ValueError("Not a valid model!")

    prompt = ChatPromptTemplate.from_template(config['prompt'])
    chain = prompt | model

    return chain, model_save_name




def get_anthropic_chain(config):
    from langchain_anthropic import ChatAnthropic
    # temp 0 to 1, defaults to 1.0
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Provide your ANTHROPIC API Key")

    if config['model_name'] == 'claude-3-sonnet-20240229':
        model_save_name = 'claude-3-sonnet'
    elif config['model_name'] == 'claude-3-opus-20240229':
        model_save_name = 'claude-3-opus'
    else:
        raise ValueError("Not a valid model!")

    model = ChatAnthropic(temperature=config["temperature"], model_name=config["model_name"])        
    prompt = ChatPromptTemplate.from_template(config['prompt'])
    chain = prompt | model

    return chain, model_save_name



def get_google_chain(config):
    # gemini-1.5-pro, gemini-1.0-pro
    # temperature:  gemini-1.5-pro: 0.0 - 2.0 (default: 1.0)
    # https://ai.google.dev/gemini-api/docs/models/generative-models
    # 
    from langchain_google_genai import ChatGoogleGenerativeAI

    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key") # ...4


    #if config['model_name'] == 'gemini-1.0-pro': outdated.
    #    model_save_name = 'gemini-1'
    #    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=config["temperature"])
    if config['model_name'] == 'gemini-1.5-flash': 
        model_save_name = 'gemini-15-flash'
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=config["temperature"])
    else:
        raise ValueError("Not a valid model!")

    prompt = ChatPromptTemplate.from_template(config['prompt'])
    chain = prompt | model

    return chain, model_save_name




def get_ollama_chain(config):

    #from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama

    #model = Ollama(model="llama3")
    model = ChatOllama(model="llama3", temperature=config["temperature"])

    if config['model_name'] == 'llama3-7b':
        model_save_name = 'llama3-7b'
    else:
        raise ValueError("Not a valid model!")

    # To do - check if ollama running?
    prompt = ChatPromptTemplate.from_template(config['prompt'])
    chain = prompt | model

    return chain, model_save_name




def get_chain(config):
    if config['model_name'] in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4' ]:
        return get_openai_chain(config)
    elif config['model_name'] in ['claude-3-sonnet-20240229','claude-3-opus-20240229']:
        return get_anthropic_chain(config)
    elif config['model_name'] in ['gemini-1.0-pro','gemini-1.5-flash']:
        return get_google_chain(config)
    elif config['model_name'] in ['Llama-3-70b-Instruct','qwen1.5-32b-chat','mixtral-8x22b-instruct']:
        return get_octo_chainl(config)
    elif config['model_name'] in ['llama3-7b']:
        return get_ollama_chain(config)
    else:
        raise ValueError("Not a valid model!")
