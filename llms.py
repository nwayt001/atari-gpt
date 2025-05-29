import json
import os
import re
import logging
import cv2
import base64
from init_models import init_model

# Set up logging
logger = logging.getLogger("atari-gpt.llms")

# Image encoding function
def encode_image(cv_image):
    _, buffer = cv2.imencode(".jpg", cv_image)
    return base64.b64encode(buffer).decode("utf-8")

# Cleaning function
def clean_model_output(output: str) -> str:
    if not output:
        return ""
    cleaned = re.sub(r'(?<!\\)\n', ' ', output)
    return cleaned


class BaseClient:
    """
    Abstract client interface. Each provider client must implement these methods.
    """
    def list_models(self) -> list:
        raise NotImplementedError

    def create_chat(self, model: str, messages: list, **kwargs):
        """Send a chat completion request"""
        raise NotImplementedError

    def extract_response_text(self, response) -> str:
        """Extract the raw text from the provider's response"""
        raise NotImplementedError


class OpenAIClient(BaseClient):
    """
    Client class for OpenAI models
    """
    def __init__(self):
        self.client = init_model()

    def list_models(self):
        """
        Function used to see if the input model name is valid
        """
        return [mdl.id.replace('models/', '') for mdl in self.client.models.list()]

    def create_chat(self, model: str, messages: list, **kwargs):
        """
        Prompts the model given the current messages for a given model.
        model: model from provider. (e.g. gpt-4.1-mini)
        messages: list of the chat history.
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )

    def extract_response_text(self, response) -> str:
        """"
        Convert the ouput from the model to a raw string.
        response: ouput from the model
        """
        return response.choices[0].message.content


class GeminiClient(OpenAIClient):
    """
    Client class for Gemini models.
    """
    def __init__(self):
        # Gemini uses the same OpenAI-compatible interface
        self.client = init_model(gemini=True)


class ClaudeClient(BaseClient):
    """
    Client class for Claude models.
    """
    def __init__(self):
        self.client = init_model(claude=True)

    def list_models(self):
        """
        Function used to see if the input model name is valid
        """
        return [mdl for mdl in self.client.list_models()]

    def create_chat(self, model: str, messages: list, **kwargs):
        """
        Prompts the model given the current messages for a given model.
        model: model from provider. (e.g. claude-3-opus-20240229)
        messages: list of the chat history.
        """
        return self.client.messages.create(
            model=model,
            messages=messages,
            **kwargs
        )

    def extract_response_text(self, response) -> str:
        """"
        Convert the ouput from the model to a raw string.
        response: ouput from the model
        """
        return response.content[0].text


# Function to get the correct provider API
def get_client(provider: str) -> BaseClient:
    provider = provider.lower()
    if provider == 'openai':
        return OpenAIClient()
    elif provider == 'gemini':
        return GeminiClient()
    elif provider == 'claude':
        return ClaudeClient()
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class Agent:
    def __init__(
        self,
        provider: str,
        model_name: str,
        system_prompt: str,
        env
    ):
        """
        Initialize the Agent with the specified provider, model name, system prompt and environment.
        
        Args:
            provider: The model provider key ('openai', 'claude', 'gemini')
            model_name: The specific model name to use
            system_prompt: The system prompt to use
            env: The Gymnasium environment
        """
        self.client = get_client(provider)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.env = env
        self.action_space = env.action_space.n
        self.messages = []
        self.provider = provider

        # If using the OpenAI API then append the system prompt to context
        if provider in ['openai', 'gemini']:
            self.messages.append({"role": "system", "content": system_prompt})

        # verify model availability
        available = self.client.list_models()
        if model_name not in available:
            raise ValueError(f"Model '{model_name}' not available. Choose from: {available}")
        logger.info(f"Using model {model_name} on {provider}")

    # Function for querying the LLM
    def query_llm(self, **kwargs):
        resp = self.client.create_chat(
            model=self.model_name,
            messages=self.messages,
            **kwargs
        )
        return resp

    def get_response(self):
        # simple retry logic
        for attempt in range(3):
            try:
                # If using Claude then send the system prompt as a kwarg
                if isinstance(self.client, ClaudeClient): 
                    resp = self.query_llm(temperature=0, system=self.system_prompt)
                    text = self.client.extract_response_text(resp)
                else:
                    resp = self.query_llm(temperature=0)
                    text = self.client.extract_response_text(resp)
                return text
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                continue
        raise RuntimeError("Failed to get response from LLM after retries")

    def generate_response(self, path: str):
        raw = self.get_response()
        cleaned = clean_model_output(raw)

        # save for debugging
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(os.path.join(path, 'all_responses.txt'), 'a') as f:
            f.write(cleaned + "\n\n")

        # parse JSON
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        payload = match.group(0) if match else cleaned
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            raise

        action = int(obj.get('action', 0))
        if action < 0 or action >= self.action_space:
            raise ValueError(f"Invalid action: {action}")

        return action, obj.get('reasoning', None)

    def add_user_message(self, text: str = None, frame=None):
        content = []
        if text:
            content.append({"type": "text", "text": text})
        if frame is not None:
            img = encode_image(frame)
            # Send images in for Claude as needed
            if isinstance(self.client, ClaudeClient):
                content.append({"type": "image", "source": {"type": "base64",  "media_type": "image/jpeg", "data": img}})
            else:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        
        self.messages.append({"role": "user", "content": content if len(content) > 1 else content[0]})

    def add_assistant_message(self, text: str):
        self.messages.append({"role": "assistant", "content": text})
    
    def delete_messages(self):

        if self.provider in ['openai', 'gemini']:
            message_len = 9
        else:
            message_len = 8

        
        if len(self.messages) >= message_len:

            if self.messages[0]['role'] == 'system':
                # Delete user message
                value = self.messages.pop(1)

                # Delete Assistant message
                value = self.messages.pop(1)

            else:
                # Delete user message
                self.messages.pop(0)

                # Delete Assistant message
                self.messages.pop(0)
