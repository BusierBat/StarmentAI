
import requests
import time

URL = "http://localhost:11434/api/chat"
MODEL = "starmentai"

THREADS = [1, 2, 4]

PROMPT = "Mày là ai? Trả lời ngắn gọn."

for thread_count in THREADS:
    print(f"\n===== TEST {thread_count} THREADS =====")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT}
        ],
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {
            "num_ctx": 1024,
            "num_thread": thread_count,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    try:
        start = time.perf_counter()

        response = requests.post(
            URL,
            json=payload,
            timeout=300
        )

        elapsed = time.perf_counter() - start
        response.raise_for_status()

        data = response.json()

        eval_count = data.get("eval_count", 0)
        eval_time = data.get("eval_duration", 0) / 1e9

        prompt_count = data.get("prompt_eval_count", 0)
        prompt_time = data.get("prompt_eval_duration", 0) / 1e9

        print(f"Total: {elapsed:.2f}s")
        print(f"Prompt: {prompt_time:.2f}s")
        print(f"Generation: {eval_time:.2f}s")
        print(f"Prompt tokens: {prompt_count}")
        print(f"Generated tokens: {eval_count}")

        if eval_time > 0:
            print(
                f"Generation speed: "
                f"{eval_count / eval_time:.2f} tokens/s"
            )

    except requests.RequestException as error:
        print(f"ERROR: {error}")

print("")