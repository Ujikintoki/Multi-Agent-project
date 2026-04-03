import os
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

# Load environment variables from .env file
load_dotenv()


class LLMClient:
    """
    A robust LLM client wrapper handling API requests, retries, and error logging.
    Designed as the foundational communication layer for all Agents.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", max_retries: int = 3):
        """
        Initialize the LLM client.

        Args:
            model_name: The target LLM model to use (e.g., 'gpt-3.5-turbo', 'gpt-4').
            max_retries: Maximum number of retry attempts for failed API calls.
        """
        # Read API key securely from environment
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the .env file.")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.max_retries = max_retries

    def generate_response(
        self, messages: List[Dict[str, str]], temperature: float = 0.2
    ) -> Optional[str]:
        """
        Send a chat completion request to the LLM with built-in retry logic.

        Args:
            messages: A list of message dictionaries (role, content).
            temperature: Sampling temperature. Lower values (e.g., 0.2) are better
                         for deterministic code generation.

        Returns:
            The generated text response from the model, or None if all retries fail.
        """
        attempt = 0

        while attempt < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    # We can enforce stop tokens or max_tokens here if needed for the baseline
                )

                # Extract and return the raw string content
                return response.choices[0].message.content

            except RateLimitError:
                attempt += 1
                wait_time = 2**attempt  # Exponential backoff: 2s, 4s, 8s...
                print(
                    f"[Warning] Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt}/{self.max_retries})"
                )
                time.sleep(wait_time)

            except APITimeoutError:
                attempt += 1
                print(
                    f"[Warning] API Timeout. Retrying in 2 seconds... (Attempt {attempt}/{self.max_retries})"
                )
                time.sleep(2)

            except APIError as e:
                print(f"[Error] OpenAI API returned an API Error: {e}")
                # For critical API errors (e.g., invalid key), we might not want to retry
                break

            except Exception as e:
                print(f"[Critical Error] An unexpected error occurred: {e}")
                break

        print("[Error] Max retries reached or fatal error encountered. Returning None.")
        return None
