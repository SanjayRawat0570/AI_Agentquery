from fastapi.testclient import TestClient
import json
import os
import sys

# Ensure project dir is on sys.path
sys.path.insert(0, os.getcwd())

from main import app
client = TestClient(app)
print('health ->', client.get('/health').json())

queries = [
    'What is the monthly price for CM-Pro and what features does it include?',
    "We're getting error E1234 when calling the API. What should we check?",
    'Compare cm-pro vs cm-enterprise; also what\'s the monthly price for cm-enterprise?',
    'How many user slots are available on ACC-1111?'
]
for q in queries:
    resp = client.post('/api/query', json={'query': q})
    print('\nQuery:', q)
    try:
        print('Status:', resp.status_code)
        print('Response JSON:', json.dumps(resp.json(), indent=2))
    except Exception as e:
        print('Error parsing response:', e)
