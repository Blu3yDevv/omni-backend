from gradio_client import Client
import json

def main():
    print("[TEST] Creating client...")
    client = Client("Blu3yDevv/omni-nano-inference")
    print("[TEST] Client loaded.")

    messages = [
        {"role": "system", "content": "You are Omni Nano. Answer in ONE short sentence."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    messages_json = json.dumps({"messages": messages})

    print("[TEST] Calling predict (max_new_tokens=32)...")
    result = client.predict(
        messages_json,
        32,    # max_new_tokens small so it’s fast
        0.1,   # low temperature
        api_name="/predict",
    )
    print("[TEST] Result:", repr(result))

if __name__ == "__main__":
    main()
