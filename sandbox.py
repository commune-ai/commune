
import requests
response = requests.post(
    "https://search.dria.co/search",
    headers={'x-api-key': '63f19480-3f06-4e7c-8c12-6d2a8c30678b', 'Content-Type': 'application/json'},
    json={
    "rerank": True,
    "top_n": 10,
    "contract_id": "7LUEmgCw9f3DJ6a0haw7mFgzNhbwdWSdWaSqrVkB1E0",
    "query": "What is alexanDRIA library?"
}
)
print(response.status_code)
print(response.json())