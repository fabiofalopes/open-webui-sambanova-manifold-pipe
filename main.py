"""
title: Samba nova system manifold api
version: 0.2.0
updated_on: 2025-01-02
reference: "https://community.sambanova.ai/t/supported-models/193"
funding_url: https://github.com/open-webui
url: https://github.com/fabiofalopes/open-webui-sambanova-manifold-pipe
community: https://openwebui.com/f/fabiofalopes/sambanova/
"""

import os
import requests
import json
import time
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    class Valves(BaseModel):
        SAMBANOVA_API_KEY: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "sambanova"
        self.name = "sambanova/"
        self.valves = self.Valves(
            **{"SAMBANOVA_API_KEY": os.getenv("SAMBANOVA_API_KEY", "")}
        )
        pass

    def get_sambanova_models(self):
        # Updated list of SambaNova models with context lengths added to display names (02-01-2025) 
        return [
            # Qwen 2.5 family
            {"id": "Qwen2.5-Coder-32B-Instruct", "name": "Qwen2.5 Coder 32B (8k)"},
            {"id": "Qwen2.5-72B-Instruct", "name": "Qwen2.5 72B (8k)"},
            {"id": "QwQ-32B-Preview", "name": "QwQ 32B Preview (8k)"},
            # Llama 3.3 family
            {"id": "Meta-Llama-3.3-70B-Instruct", "name": "Llama 3.3 70B (4k)"},
            # Llama 3.2 family
            {"id": "Meta-Llama-3.2-1B-Instruct", "name": "Llama 3.2 1B (16k)"},
            {"id": "Meta-Llama-3.2-3B-Instruct", "name": "Llama 3.2 3B (4k)"},
            {"id": "Llama-3.2-11B-Vision-Instruct", "name": "Llama 3.2 11B (4k)"},
            {"id": "Llama-3.2-90B-Vision-Instruct", "name": "Llama 3.2 90B (4k)"},
            # Llama 3.1 family
            {"id": "Meta-Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B (16k)"},
            {"id": "Meta-Llama-3.1-70B-Instruct", "name": "Llama 3.1 70B (128k)"},
            {"id": "Meta-Llama-3.1-405B-Instruct", "name": "Llama 3.1 405B (16k)"},
            {"id": "Meta-Llama-Guard-3-8B", "name": "Llama Guard 3 8B (8k)"},
        ]

    def pipes(self) -> List[dict]:
        return self.get_sambanova_models()

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        # Handle system messages if present (assumes system messages are separated)
        system_message, messages = pop_system_message(body["messages"])

        processed_messages = []
        for message in messages:
            processed_messages.append(
                {"role": message["role"], "content": message.get("content", "")}
            )

        # Remove the "samba_nova_api." prefix if present in the model ID
        model_id = body["model"]
        if model_id.startswith("sambanova."):
            model_id = model_id[len("sambanova.") :]
        elif model_id.startswith("samba_nova_api."):
            model_id = model_id[len("samba_nova_api.") :]

        payload = {
            "model": model_id,
            "messages": processed_messages,
            "max_tokens": body.get("max_tokens", 4096),
            "temperature": body.get("temperature", 0.8),
            "top_k": body.get("top_k", 40),
            "top_p": body.get("top_p", 0.9),
            "stop": body.get("stop", []),
            "stream": body.get("stream", False),
        }

        headers = {
            "Authorization": f"Bearer {self.valves.SAMBANOVA_API_KEY}",
            "Content-Type": "application/json",
        }

        url = "https://api.sambanova.ai/v1/chat/completions"

        try:
            if body.get("stream", False):
                return self.stream_response(url, headers, payload)
            else:
                return self.non_stream_response(url, headers, payload)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        try:
            with requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(3.05, 60),
                verify=False,
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data["choices"][0]["delta"].get("content"):
                                    yield data["choices"][0]["delta"]["content"]

                                time.sleep(0.01)
                            except json.JSONDecodeError:
                                print(f"Failed to parse JSON: {line}")
                            except KeyError as e:
                                print(f"Unexpected data structure: {e}")
                                print(f"Full data: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            yield f"Error: Request failed: {e}"
        except Exception as e:
            print(f"General error in stream_response method: {e}")
            yield f"Error: {e}"

    def non_stream_response(self, url, headers, payload):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=(3.05, 60), verify=False
            )
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            res = response.json()
            return res["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"

