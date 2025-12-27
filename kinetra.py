"""
Kinetra - OpenRouter Integration

This module provides a simple interface to interact with OpenRouter API.
OpenRouter is a unified interface for accessing various LLM models.
"""

import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class OpenRouterClient:
    """Client for interacting with OpenRouter API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If not provided, will try to load from OPENROUTER_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-3.5-turbo",
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict:
        """
        Create a chat completion using OpenRouter.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model identifier (e.g., 'openai/gpt-3.5-turbo', 'anthropic/claude-2')
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0-2)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response as a dictionary
            
        Example:
            >>> client = OpenRouterClient()
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> response = client.chat_completion(messages)
            >>> print(response['choices'][0]['message']['content'])
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def list_models(self) -> List[Dict]:
        """
        Get list of available models from OpenRouter.
        
        Returns:
            List of model dictionaries with information about available models
        """
        endpoint = f"{self.base_url}/models"
        
        response = requests.get(endpoint, headers=self.headers)
        response.raise_for_status()
        
        return response.json()


def main():
    """Example usage of the OpenRouter client"""
    try:
        # Initialize the client
        client = OpenRouterClient()
        
        # Example: Simple chat completion
        print("Sending a test message to OpenRouter...")
        messages = [
            {"role": "user", "content": "Say hello and introduce yourself briefly."}
        ]
        
        response = client.chat_completion(
            messages=messages,
            model="openai/gpt-3.5-turbo",
            max_tokens=100
        )
        
        # Extract and print the response
        assistant_message = response['choices'][0]['message']['content']
        print(f"\nAssistant: {assistant_message}")
        
        # Print usage information
        if 'usage' in response:
            print(f"\nTokens used: {response['usage'].get('total_tokens', 'N/A')}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease set your OpenRouter API key:")
        print("1. Copy .env.example to .env")
        print("2. Add your API key from https://openrouter.ai/keys")
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
