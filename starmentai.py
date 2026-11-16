import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "starmentai",
        "messages": [{"role": "user", "content": "Xin chào!"}],
        "stream": False,
        "think": False,
    },
    timeout=60,
)

print(response.json()["message"]["content"])
