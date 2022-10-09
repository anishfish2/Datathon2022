import requests
import json
import re

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



# Define your API key here 
API_KEY = "IJXH6TU5QL9BFnRJHCl8G99pKBFkTIMt6smwp0cU"
with open("input.txt") as f:
    TEXT = f.read()
    pattern = re.compile('[\W_]+')
    pattern.sub('', TEXT)

# Call the get_embedding function located in ./assets
embedding = get_embedding(TEXT, API_KEY)

# Print stuff out and be prepared for a ton of numbers
print(embedding)

