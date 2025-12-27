"""
Example usage of Kinetra OpenRouter integration

This script demonstrates different ways to use the OpenRouter API.
"""

from kinetra import OpenRouterClient


def example_simple_chat():
    """Example: Simple chat interaction"""
    print("=" * 60)
    print("Example 1: Simple Chat")
    print("=" * 60)
    
    client = OpenRouterClient()
    
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = client.chat_completion(messages)
    print(f"Q: {messages[0]['content']}")
    print(f"A: {response['choices'][0]['message']['content']}\n")


def example_conversation():
    """Example: Multi-turn conversation"""
    print("=" * 60)
    print("Example 2: Multi-turn Conversation")
    print("=" * 60)
    
    client = OpenRouterClient()
    
    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What's my name?"}
    ]
    
    response = client.chat_completion(messages)
    print(f"Conversation history: {len(messages)} messages")
    print(f"Response: {response['choices'][0]['message']['content']}\n")


def example_different_models():
    """Example: Using different models"""
    print("=" * 60)
    print("Example 3: Using Different Models")
    print("=" * 60)
    
    client = OpenRouterClient()
    
    messages = [
        {"role": "user", "content": "Say hi in one sentence."}
    ]
    
    # Try with different models (you can change these based on your preference)
    models = [
        "openai/gpt-3.5-turbo",
        "anthropic/claude-2",
        "meta-llama/llama-2-70b-chat"
    ]
    
    for model in models:
        try:
            response = client.chat_completion(messages, model=model)
            print(f"\nModel: {model}")
            print(f"Response: {response['choices'][0]['message']['content']}")
        except Exception as e:
            print(f"\nModel: {model}")
            print(f"Error: {e}")


def example_list_models():
    """Example: List available models"""
    print("=" * 60)
    print("Example 4: List Available Models")
    print("=" * 60)
    
    client = OpenRouterClient()
    
    try:
        models = client.list_models()
        print(f"\nTotal models available: {len(models.get('data', []))}")
        print("\nSample models:")
        for model in models.get('data', [])[:5]:
            print(f"  - {model.get('id', 'N/A')}")
    except Exception as e:
        print(f"Error listing models: {e}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("KINETRA - OpenRouter Integration Examples")
    print("=" * 60 + "\n")
    
    try:
        # Run examples
        example_simple_chat()
        example_conversation()
        # Uncomment to try different models (may incur costs)
        # example_different_models()
        # example_list_models()
        
        print("=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("\nSetup Instructions:")
        print("1. Copy .env.example to .env")
        print("2. Get your API key from https://openrouter.ai/keys")
        print("3. Add your API key to the .env file")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
