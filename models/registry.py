from models.openai_adapter import OpenAIAdapter
from models.anthropic_adapter import AnthropicAdapter
from models.gemini_adapter import GeminiAdapter


PROVIDER_MAP = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "google": GeminiAdapter,
}


def get_model(config):

    provider = config["provider"]
    model_name = config["model_name"]

    adapter_class = PROVIDER_MAP[provider]

    return adapter_class(model_name)