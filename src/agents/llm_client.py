# src/agents/llm_client.py
import asyncio
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, AsyncAzureOpenAI, RateLimitError

load_dotenv()


class AsyncLLMClient:
    """
    An asynchronous Azure LLM client wrapper handling API requests, retries, and error logging.
    """

    def __init__(self, max_retries: int = 3):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

        if not all([self.api_key, self.endpoint, self.api_version, self.deployment]):
            raise ValueError("Missing required Azure OpenAI credentials in .env file.")

        default_headers = {"Ocp-Apim-Subscription-Key": self.api_key}

        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint.rstrip("/"),
            api_version=self.api_version,
            default_headers=default_headers,
            timeout=60.0,
        )
        self.max_retries = max_retries

    async def generate_response(
        self, messages: List[Dict[str, str]], temperature: float = 1.0
    ) -> Optional[str]:
        """Send an async chat completion request with built-in retry logic."""
        attempt = 0

        while attempt < self.max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content

            except RateLimitError:
                attempt += 1
                wait_time = 2**attempt
                print(
                    f"[Warning] Rate limit hit. Retrying in {wait_time}s... ({attempt}/{self.max_retries})"
                )
                await asyncio.sleep(wait_time)

            except APITimeoutError:
                attempt += 1
                print(
                    f"[Warning] API Timeout. Retrying in 2s... ({attempt}/{self.max_retries})"
                )
                await asyncio.sleep(2)

            except APIError as e:
                print(f"[Error] Azure API Error: {e}")
                break

            except Exception as e:
                print(f"[Critical Error] Unexpected error: {e}")
                break

        print("[Error] Max retries reached or fatal error. Returning None.")
        return None
        return None
        return None
