import os
import requests
import json

# #Set a API_TOKEN environment variable before running
API_TOKEN = os.environ["API_TOKEN"]
API_URL =  "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2" #"https://api-inference.huggingface.co/models/ESGBERT/EnvironmentalBERT-environmental" #Add a URL for a model of your choosing
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(prompt_question, prompt_context, api_url=API_URL, headers=headers):
    prompt = {
        "question": prompt_question,
        "context": prompt_context,
    }

    payload = {
        "inputs": prompt,
        "parameters": { #Try and experiment with the parameters
            "max_new_tokens": 1024,
            "temperature": 0.9,
            "top_p": 0.9,
            "do_sample": False,
            "return_full_text": False,
            "wait_for_model": True,
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json() #.json()[0]['generated_text']

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('***** LLM Demo - Inference API ***')

question = "What is the population of Miami, Florida?"
context = "As of the most current census, Jacksonville, Florida has a population of 1.13 million."

print("**** Roberta Model **** ")
print(query(question, context))
print("**** Bert Large Model **** ")
print(query(question,context,"https://api-inference.huggingface.co/models/valhalla/bart-large-finetuned-squadv1"))
print("**** Llama2 7b **** ")
print(query(question,context,"https://api-inference.huggingface.co/models/cxllin/Llama2-7b-med-v1"))
