import requests
import json
from scipy import spatial
import pandas as pd

def get_embedding(text, api_key):
    ## API Definitions
    url = "https://datathon.bindgapi.com/channel"
    headers =  {
        "X-API-Key": api_key,
        "Content-Type":"application/json"
    }
    body = { "inputs": text }
    ## API Call
    try:
        response = requests.post(url, data=json.dumps(body), headers=headers)
    except Exception:
        print(Exception)
        
    try:
        # return response 
        result = response.json()
        return json.loads(result['results'])
    except:
        print(response.status_code)

def similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)


# Define your API key here 
API_KEY = "IJXH6TU5QL9BFnRJHCl8G99pKBFkTIMt6smwp0cU"
with open("api_input.txt") as f:
    TEXT = f.read()[:7500]

# Call the get_embedding function located in ./assets
embd1 = get_embedding(TEXT, API_KEY)

challenge = pd.read_csv('challenge.csv')
num = 1
embd0 = [float(x) for x in challenge['embeddings'][num][1:-1].split(", ")]
print(f'Challenge: {num}  {similarity(embd0, embd1)}')
