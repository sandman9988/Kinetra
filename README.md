# Kinetra - OpenRouter Integration

A simple and elegant Python interface for interacting with [OpenRouter](https://openrouter.ai/), providing unified access to multiple Large Language Model (LLM) APIs.

## What is OpenRouter?

OpenRouter is a unified API that provides access to various LLM models from different providers (OpenAI, Anthropic, Meta, Google, etc.) through a single interface. This means you can:

- Access multiple AI models with one API key
- Switch between models easily
- Compare responses from different models
- Avoid vendor lock-in

## Features

- 🚀 Simple and intuitive API
- 🔑 Easy API key management with environment variables
- 💬 Chat completion support
- 🎯 Multiple model support
- 📝 Comprehensive examples
- 🔒 Secure credential handling

## Installation

1. Clone this repository:
```bash
git clone https://github.com/sandman9988/Kinetra.git
cd Kinetra
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# Get your key from: https://openrouter.ai/keys
```

## Quick Start

### Basic Usage

```python
from kinetra import OpenRouterClient

# Initialize the client
client = OpenRouterClient()

# Send a chat message
messages = [
    {"role": "user", "content": "Hello! How are you?"}
]

response = client.chat_completion(messages)
print(response['choices'][0]['message']['content'])
```

### Run the Example Script

```bash
python kinetra.py
```

### Explore More Examples

```bash
python examples.py
```

## API Reference

### OpenRouterClient

#### `__init__(api_key: Optional[str] = None, timeout: int = 30)`

Initialize the OpenRouter client.

**Parameters:**
- `api_key` (optional): Your OpenRouter API key. If not provided, it will be read from the `OPENROUTER_API_KEY` environment variable.
- `timeout` (int): Request timeout in seconds (default: 30)

#### `chat_completion(messages, model, max_tokens, temperature, **kwargs)`

Create a chat completion.

**Parameters:**
- `messages` (List[Dict]): List of message dictionaries with 'role' and 'content' keys
- `model` (str): Model identifier (default: "openai/gpt-3.5-turbo")
  - Examples: "openai/gpt-4", "anthropic/claude-2", "meta-llama/llama-2-70b-chat"
- `max_tokens` (int, optional): Maximum tokens in the response
- `temperature` (float): Sampling temperature 0-2 (default: 1.0)
- `**kwargs`: Additional parameters supported by OpenRouter API

**Returns:** API response dictionary

#### `list_models()`

Get a list of available models from OpenRouter.

**Returns:** List of model dictionaries

## Available Models

OpenRouter provides access to models from various providers:

- **OpenAI**: GPT-4, GPT-3.5-Turbo, etc.
- **Anthropic**: Claude 2, Claude Instant
- **Meta**: Llama 2 (various sizes)
- **Google**: PaLM 2
- And many more!

Check the [OpenRouter models page](https://openrouter.ai/models) for the full list.

## Examples

### Simple Question

```python
from kinetra import OpenRouterClient

client = OpenRouterClient()
messages = [{"role": "user", "content": "What is Python?"}]
response = client.chat_completion(messages)
print(response['choices'][0]['message']['content'])
```

### Multi-turn Conversation

```python
messages = [
    {"role": "user", "content": "My favorite color is blue."},
    {"role": "assistant", "content": "That's great! Blue is a lovely color."},
    {"role": "user", "content": "What's my favorite color?"}
]

response = client.chat_completion(messages)
```

### Using Different Models

```python
# Use GPT-4
response = client.chat_completion(
    messages,
    model="openai/gpt-4"
)

# Use Claude
response = client.chat_completion(
    messages,
    model="anthropic/claude-2"
)
```

### Adjusting Parameters

```python
response = client.chat_completion(
    messages,
    model="openai/gpt-3.5-turbo",
    max_tokens=150,
    temperature=0.7
)
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### Getting Your API Key

1. Go to [OpenRouter](https://openrouter.ai/)
2. Sign up or log in
3. Navigate to [API Keys](https://openrouter.ai/keys)
4. Generate a new API key
5. Add it to your `.env` file

## Security

⚠️ **Important Security Notes:**

- Never commit your `.env` file or API keys to version control
- The `.gitignore` file is configured to exclude `.env` files
- Use environment variables for API keys in production
- Rotate your API keys regularly
- In production environments, consider using a secrets management service (e.g., AWS Secrets Manager, HashiCorp Vault) instead of plain environment variables
- API keys are stored in memory during runtime; ensure your environment is secure

## Requirements

- Python 3.7+
- requests>=2.31.0
- python-dotenv>=1.0.0

## Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest features
- Submit pull requests

## License

This project is open source and available under the MIT License.

## Resources

- [OpenRouter Website](https://openrouter.ai/)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter API Keys](https://openrouter.ai/keys)
- [Available Models](https://openrouter.ai/models)

## Support

For issues or questions:
- Open an issue on GitHub
- Check the [OpenRouter Discord](https://discord.gg/openrouter)

---

**Made with ❤️ for the AI community**
