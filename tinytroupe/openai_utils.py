import os
import openai
from openai import OpenAI, AzureOpenAI
import time
import pickle
import logging
import configparser
import requests
import json
from typing import Union


import tiktoken
from tinytroupe import utils
from tinytroupe.control import transactional
from tinytroupe import default
from tinytroupe import config_manager

logger = logging.getLogger("tinytroupe")

# We'll use various configuration elements below
config = utils.read_config_file()

###########################################################################
# Client class
###########################################################################

class OpenAIClient:
    """
    A utility class for interacting with the OpenAI API.
    """

    def __init__(self, cache_api_calls=default["cache_api_calls"], cache_file_name=default["cache_file_name"]) -> None:
        logger.debug("Initializing OpenAIClient")

        # should we cache api calls and reuse them?
        self.set_api_cache(cache_api_calls, cache_file_name)
    
    def set_api_cache(self, cache_api_calls, cache_file_name=default["cache_file_name"]):
        """
        Enables or disables the caching of API calls.

        Args:
        cache_file_name (str): The name of the file to use for caching API calls.
        """
        self.cache_api_calls = cache_api_calls
        self.cache_file_name = cache_file_name
        if self.cache_api_calls:
            # load the cache, if any
            self.api_cache = self._load_cache()
    
    
    def _setup_from_config(self):
        """
        Sets up the OpenAI API configurations for this client.
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @config_manager.config_defaults(
        model="model",
        temperature="temperature",
        max_tokens="max_tokens",
        top_p="top_p",
        frequency_penalty="frequency_penalty",
        presence_penalty="presence_penalty",
        timeout="timeout",
        max_attempts="max_attempts",
        waiting_time="waiting_time",
        exponential_backoff_factor="exponential_backoff_factor",
        response_format=None,
        echo=None
    )
    def send_message(self,
                    current_messages,
                    dedent_messages=True,
                    model=None,
                    temperature=None,
                    max_tokens=None,
                    top_p=None,
                    frequency_penalty=None,
                    presence_penalty=None,
                    stop=[],
                    timeout=None,
                    max_attempts=None,
                    waiting_time=None,
                    exponential_backoff_factor=None,
                    n = 1,
                    response_format=None,
                    enable_pydantic_model_return=False,
                    echo=False):
        """
        Sends a message to the OpenAI API and returns the response.

        Args:
        current_messages (list): A list of dictionaries representing the conversation history.
        dedent_messages (bool): Whether to dedent the messages before sending them to the API.
        model (str): The ID of the model to use for generating the response.
        temperature (float): Controls the "creativity" of the response. Higher values result in more diverse responses.
        max_tokens (int): The maximum number of tokens (words or punctuation marks) to generate in the response.
        top_p (float): Controls the "quality" of the response. Higher values result in more coherent responses.
        frequency_penalty (float): Controls the "repetition" of the response. Higher values result in less repetition.
        presence_penalty (float): Controls the "diversity" of the response. Higher values result in more diverse responses.
        stop (str): A string that, if encountered in the generated response, will cause the generation to stop.
        max_attempts (int): The maximum number of attempts to make before giving up on generating a response.
        timeout (int): The maximum number of seconds to wait for a response from the API.
        waiting_time (int): The number of seconds to wait between requests.
        exponential_backoff_factor (int): The factor by which to increase the waiting time between requests.
        n (int): The number of completions to generate.
        response_format: The format of the response, if any.
        echo (bool): Whether to echo the input message in the response.
        enable_pydantic_model_return (bool): Whether to enable Pydantic model return instead of dict when possible.

        Returns:
        A dictionary representing the generated response.
        """

        def aux_exponential_backoff():
            nonlocal waiting_time

            # in case waiting time was initially set to 0
            if waiting_time <= 0:
                waiting_time = 2

            logger.info(f"Request failed. Waiting {waiting_time} seconds between requests...")
            time.sleep(waiting_time)

            # exponential backoff
            waiting_time = waiting_time * exponential_backoff_factor

        # setup the OpenAI configurations for this client.
        self._setup_from_config()

        # dedent the messages (field 'content' only) if needed (using textwrap)
        if dedent_messages:
            for message in current_messages:
                if "content" in message:
                    message["content"] = utils.dedent(message["content"])
            
        
        # We need to adapt the parameters to the API type, so we create a dictionary with them first
        chat_api_params = {
            "model": model,
            "messages": current_messages,
            "temperature": temperature,
            "max_tokens":max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "timeout": timeout,
            "stream": False,
            "n": n,
        }

        if response_format is not None:
            chat_api_params["response_format"] = response_format

        i = 0
        while i < max_attempts:
            try:
                i += 1

                try:
                    logger.debug(f"Sending messages to OpenAI API. Token count={self._count_tokens(current_messages, model)}.")
                except NotImplementedError:
                    logger.debug(f"Token count not implemented for model {model}.")
                    
                start_time = time.monotonic()
                logger.debug(f"Calling model with client class {self.__class__.__name__}.")

                ###############################################################
                # call the model, either from the cache or from the API
                ###############################################################
                cache_key = str((model, chat_api_params)) # need string to be hashable
                if self.cache_api_calls and (cache_key in self.api_cache):
                    response = self.api_cache[cache_key]
                else:
                    if waiting_time > 0:
                        logger.info(f"Waiting {waiting_time} seconds before next API request (to avoid throttling)...")
                        time.sleep(waiting_time)
                    
                    response = self._raw_model_call(model, chat_api_params)
                    if self.cache_api_calls:
                        self.api_cache[cache_key] = response
                        self._save_cache()
                
                
                logger.debug(f"Got response from API: {response}")
                end_time = time.monotonic()
                logger.debug(
                    f"Got response in {end_time - start_time:.2f} seconds after {i} attempts.")

                if enable_pydantic_model_return:
                    return utils.to_pydantic_or_sanitized_dict(self._raw_model_response_extractor(response), model=response_format)
                else:
                    return utils.sanitize_dict(self._raw_model_response_extractor(response))

            except InvalidRequestError as e:
                logger.error(f"[{i}] Invalid request error, won't retry: {e}")

                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None
            
            except openai.BadRequestError as e:
                logger.error(f"[{i}] Invalid request error, won't retry: {e}")
                
                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None
            
            except openai.RateLimitError:
                logger.warning(
                    f"[{i}] Rate limit error, waiting a bit and trying again.")
                aux_exponential_backoff()
            
            except NonTerminalError as e:
                logger.error(f"[{i}] Non-terminal error: {e}")
                aux_exponential_backoff()
                
            except Exception as e:
                logger.error(f"[{i}] {type(e).__name__} Error: {e}")
                aux_exponential_backoff()

        logger.error(f"Failed to get response after {max_attempts} attempts.")
        return None
    
    def _raw_model_call(self, model, chat_api_params):
        """
        Calls the OpenAI API with the given parameters. Subclasses should
        override this method to implement their own API calls.
        """   

        # adjust parameters depending on the model
        if self._is_reasoning_model(model):
            # Reasoning models have slightly different parameters
            del chat_api_params["stream"]
            del chat_api_params["temperature"]
            del chat_api_params["top_p"]
            del chat_api_params["frequency_penalty"]
            del chat_api_params["presence_penalty"]            

            chat_api_params["max_completion_tokens"] = chat_api_params["max_tokens"]
            del chat_api_params["max_tokens"]

            chat_api_params["reasoning_effort"] = default["reasoning_effort"]


        # To make the log cleaner, we remove the messages from the logged parameters
        logged_params = {k: v for k, v in chat_api_params.items() if k != "messages"} 

        if "response_format" in chat_api_params:
            # to enforce the response format via pydantic, we need to use a different method

            if "stream" in chat_api_params:
                del chat_api_params["stream"]

            logger.debug(f"Calling LLM model (using .parse too) with these parameters: {logged_params}. Not showing 'messages' parameter.")
            # complete message
            logger.debug(f"   --> Complete messages sent to LLM: {chat_api_params['messages']}")

            result_message = self.client.beta.chat.completions.parse(
                    **chat_api_params
                )

            return result_message 
        
        else:
            logger.debug(f"Calling LLM model with these parameters: {logged_params}. Not showing 'messages' parameter.")
            return self.client.chat.completions.create(
                        **chat_api_params
                    )

    def _is_reasoning_model(self, model):
        return "o1" in model or "o3" in model

    def _raw_model_response_extractor(self, response):
        """
        Extracts the response from the API response. Subclasses should
        override this method to implement their own response extraction.
        """
        return response.choices[0].message.to_dict()

    def _count_tokens(self, messages: list, model: str):
        """
        Count the number of OpenAI tokens in a list of messages using tiktoken.

        Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        Args:
        messages (list): A list of dictionaries representing the conversation history.
        model (str): The name of the model to use for encoding the string.
        """
        try:
            # For Ollama models, use a simple estimation since tiktoken won't work
            if (config["OpenAI"].get("API_TYPE") == "ollama" or 
                (not any(m in model.lower() for m in ["gpt", "openai", "azure", "o1", "o3", "ppo"]) and 
                 any(m in model.lower() for m in ["llama", "mistral", "qwen", "mixtral", "phi", "gemma", "ollama"]))):
                
                # For Ollama models, estimate tokens based on a simple 4 chars per token rule
                logger.debug(f"Using estimated token count for Ollama model: {model}")
                
                total_chars = sum(len(message.get("content", "")) for message in messages)
                # Roughly 4 characters per token for most languages
                return max(1, int(total_chars / 4))
                
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                logger.debug("Token count: model not found. Using cl100k_base encoding.")
                encoding = tiktoken.get_encoding("cl100k_base")
            
            if model in {
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
                "gpt-4-0314",
                "gpt-4-32k-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
                } or "o1" in model or "o3" in model: # assuming o1/3 models work the same way
                tokens_per_message = 3
                tokens_per_name = 1
            elif model == "gpt-3.5-turbo-0301":
                tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
                tokens_per_name = -1  # if there's a name, the role is omitted
            elif "gpt-3.5-turbo" in model:
                logger.debug("Token count: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
                return self._count_tokens(messages, model="gpt-3.5-turbo-0613")
            elif ("gpt-4" in model) or ("ppo" in model) :
                logger.debug("Token count: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
                return self._count_tokens(messages, model="gpt-4-0613")
            else:
                # For unknown models, use a simple estimation
                logger.debug(f"Using cl100k_base encoding for unknown model: {model}")
                tokens_per_message = 3
                tokens_per_name = 1
            
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
            return num_tokens
        
        except Exception as e:
            # If token counting fails, estimate based on character count
            logger.warning(f"Error counting tokens with tiktoken: {e}")
            logger.warning("Using character-based estimation instead")
            
            total_chars = sum(len(message.get("content", "")) for message in messages)
            # Roughly 4 characters per token for most languages
            return max(1, int(total_chars / 4))

    def _save_cache(self):
        """
        Saves the API cache to disk. We use pickle to do that because some obj
        are not JSON serializable.
        """
        # use pickle to save the cache
        pickle.dump(self.api_cache, open(self.cache_file_name, "wb", encoding="utf-8", errors="replace"))

    
    def _load_cache(self):

        """
        Loads the API cache from disk.
        """
        # unpickle
        return pickle.load(open(self.cache_file_name, "rb", encoding="utf-8", errors="replace")) if os.path.exists(self.cache_file_name) else {}

    def get_embedding(self, text, model=default["embedding_model"]):
        """
        Gets the embedding of the given text using the specified model.

        Args:
        text (str): The text to embed.
        model (str): The name of the model to use for embedding the text.

        Returns:
        The embedding of the text.
        """
        response = self._raw_embedding_model_call(text, model)
        return self._raw_embedding_model_response_extractor(response)
    
    def _raw_embedding_model_call(self, text, model):
        """
        Calls the OpenAI API to get the embedding of the given text. Subclasses should
        override this method to implement their own API calls.
        """
        return self.client.embeddings.create(
            input=[text],
            model=model
        )
    
    def _raw_embedding_model_response_extractor(self, response):
        """
        Extracts the embedding from the API response. Subclasses should
        override this method to implement their own response extraction.
        """
        return response.data[0].embedding

class AzureClient(OpenAIClient):

    def __init__(self, cache_api_calls=default["cache_api_calls"], cache_file_name=default["cache_file_name"]) -> None:
        logger.debug("Initializing AzureClient")

        super().__init__(cache_api_calls, cache_file_name)
    
    def _setup_from_config(self):
        """
        Sets up the Azure OpenAI Service API configurations for this client,
        including the API endpoint and key.
        """
        if os.getenv("AZURE_OPENAI_KEY"):
            logger.info("Using Azure OpenAI Service API with key.")
            self.client = AzureOpenAI(azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
                                    api_version = config["OpenAI"]["AZURE_API_VERSION"],
                                    api_key = os.getenv("AZURE_OPENAI_KEY"))
        else:  # Use Entra ID Auth
            logger.info("Using Azure OpenAI Service API with Entra ID Auth.")
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
            self.client = AzureOpenAI(
                azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version = config["OpenAI"]["AZURE_API_VERSION"],
                azure_ad_token_provider=token_provider
            )

class OllamaClient(OpenAIClient):
    """
    A client for interacting with Ollama API.
    """
    
    def __init__(self, cache_api_calls=default["cache_api_calls"], cache_file_name=default["cache_file_name"]) -> None:
        logger.debug("Initializing OllamaClient")
        super().__init__(cache_api_calls, cache_file_name)
        
    def _setup_from_config(self):
        """
        Sets up the Ollama API configurations for this client.
        Since Ollama doesn't use the OpenAI client library, we just store the base URL.
        """
        self.base_url = config["OpenAI"].get("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info(f"Using Ollama API at {self.base_url}")
        # We don't set self.client here as we're using direct HTTP requests
    
    def _raw_model_call(self, model, chat_api_params):
        """
        Calls the Ollama API with the given parameters.
        """
        # Extract parameters needed for Ollama API
        messages = chat_api_params.get("messages", [])
        
        logger.debug(f"Using Ollama model: {model}")
        
        # First try using the /api/chat endpoint
        try:
            # If we have multiple messages, try the chat API
            if len(messages) > 1:
                return self._try_chat_api(model, messages, chat_api_params)
            else:
                # For a single message, use the generate API which is more widely supported
                return self._try_generate_api(model, messages, chat_api_params)
        except Exception as e:
            logger.error(f"Error in primary API call method: {e}")
            # Fall back to other method if the first one failed
            try:
                if len(messages) > 1:
                    return self._try_generate_api(model, messages, chat_api_params)
                else:
                    return self._try_chat_api(model, messages, chat_api_params)
            except Exception as fallback_e:
                logger.error(f"Error in fallback API call method: {fallback_e}")
                raise NonTerminalError(f"Ollama API error: All methods failed")
    
    def _try_chat_api(self, model, messages, chat_api_params):
        """Try using the Ollama chat API endpoint"""
        # Prepare the request payload for Ollama chat API
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {}
        }
        
        # Map OpenAI parameters to Ollama parameters
        if "temperature" in chat_api_params:
            payload["options"]["temperature"] = chat_api_params["temperature"]
        
        if "max_tokens" in chat_api_params:
            payload["options"]["num_predict"] = chat_api_params["max_tokens"]
        
        if "top_p" in chat_api_params:
            payload["options"]["top_p"] = chat_api_params["top_p"]
        
        # Send the request to Ollama
        logger.debug(f"Sending request to Ollama chat API: {self.base_url}/api/chat")
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=chat_api_params.get("timeout", 60)
        )
        
        response.raise_for_status()
        
        try:
            result = response.json()
            return self._convert_to_openai_format(result)
        except Exception as json_error:
            logger.warning(f"Failed to parse JSON response: {json_error}")
            # Create a response with the raw text
            return self._create_raw_text_response(response.text)
    
    def _try_generate_api(self, model, messages, chat_api_params):
        """Try using the Ollama generate API endpoint which is more commonly supported"""
        # Extract the text from the last user message
        last_message = None
        system_prompt = None
        
        for msg in messages:
            if msg["role"] == "user":
                last_message = msg["content"]
            elif msg["role"] == "system":
                system_prompt = msg["content"]
        
        if not last_message:
            last_message = "Hello"  # Fallback
        
        # Combine system prompt with user message if both exist
        prompt = last_message
        if system_prompt:
            prompt = f"System: {system_prompt}\n\nUser: {last_message}"
        
        # Prepare payload for generate API
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {}
        }
        
        # Map OpenAI parameters to Ollama parameters
        if "temperature" in chat_api_params:
            payload["options"]["temperature"] = chat_api_params["temperature"]
        
        if "max_tokens" in chat_api_params:
            payload["options"]["num_predict"] = chat_api_params["max_tokens"]
        
        if "top_p" in chat_api_params:
            payload["options"]["top_p"] = chat_api_params["top_p"]
        
        # Send the request to Ollama
        logger.debug(f"Sending request to Ollama generate API: {self.base_url}/api/generate")
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=chat_api_params.get("timeout", 60)
        )
        
        response.raise_for_status()
        
        try:
            result = response.json()
            # Convert generate response to chat format
            chat_result = {
                "message": {
                    "content": result.get("response", ""),
                    "role": "assistant"
                }
            }
            return self._convert_to_openai_format(chat_result)
        except Exception as json_error:
            logger.warning(f"Failed to parse JSON response: {json_error}")
            # Create a response with the raw text
            return self._create_raw_text_response(response.text)
    
    def _convert_to_openai_format(self, ollama_response):
        """
        Converts Ollama API response format to OpenAI format.
        """
        # Create a response object that mimics the OpenAI response structure
        openai_format = type('obj', (object,), {
            "choices": [
                type('obj', (object,), {
                    "message": type('obj', (object,), {
                        "content": ollama_response.get("message", {}).get("content", ""),
                        "role": ollama_response.get("message", {}).get("role", "assistant"),
                        "to_dict": lambda: {
                            "content": ollama_response.get("message", {}).get("content", ""),
                            "role": ollama_response.get("message", {}).get("role", "assistant")
                        }
                    })
                })
            ]
        })
        
        return openai_format
        
    def _create_raw_text_response(self, raw_text):
        """
        Creates an OpenAI format response object from raw text when JSON parsing fails.
        """
        # Create a response object that mimics the OpenAI response structure
        openai_format = type('obj', (object,), {
            "choices": [
                type('obj', (object,), {
                    "message": type('obj', (object,), {
                        "content": raw_text,
                        "role": "assistant",
                        "to_dict": lambda: {
                            "content": raw_text,
                            "role": "assistant"
                        }
                    })
                })
            ]
        })
        
        return openai_format
    
    def _raw_embedding_model_call(self, text, model):
        """
        Calls the Ollama API to get the embedding of the given text.
        """
        # For embeddings, we'll try to be more resilient by falling back to a simpler format
        # and using a compatible model name if possible
        
        # Try first with original model
        try:
            return self._try_embedding_call(text, model)
        except Exception as e:
            logger.warning(f"Error getting embeddings with model {model}: {e}")
            
            # Fall back to a few common embedding models available in Ollama
            fallback_models = ["nomic-embed-text", "all-minilm", "bge-small"]
            
            for fallback_model in fallback_models:
                try:
                    logger.info(f"Trying fallback embedding model: {fallback_model}")
                    return self._try_embedding_call(text, fallback_model)
                except Exception as fallback_e:
                    logger.warning(f"Error with fallback model {fallback_model}: {fallback_e}")
            
            # If all fallbacks fail, create a dummy embedding as last resort
            logger.error("All embedding attempts failed, returning dummy embedding")
            import numpy as np
            dummy_embedding = np.random.normal(0, 1, 384).tolist()  # Standard size for small embeddings
            
            # Create a response object that mimics the OpenAI response structure
            return type('obj', (object,), {
                "data": [
                    type('obj', (object,), {
                        "embedding": dummy_embedding
                    })
                ]
            })
    
    def _try_embedding_call(self, text, model):
        """Try to get an embedding from Ollama with a specific model"""
        payload = {
            "model": model,
            "prompt": text
        }
        
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Create a response object that mimics the OpenAI response structure
        return type('obj', (object,), {
            "data": [
                type('obj', (object,), {
                    "embedding": result.get("embedding", [])
                })
            ]
        })
            
class OllamaClient(OpenAIClient):
    """
    A client for interacting with Ollama API.
    This client implements the same interface as OpenAIClient but uses the Ollama API instead.
    """

    def __init__(self, cache_api_calls=default["cache_api_calls"], cache_file_name=default["cache_file_name"]) -> None:
        logger.debug("Initializing OllamaClient")
        super().__init__(cache_api_calls, cache_file_name)
        self.base_url = config["OpenAI"].get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    def _setup_from_config(self):
        """
        Sets up the Ollama API configurations for this client.
        Unlike OpenAI client, Ollama doesn't need an API key and uses a simple REST API.
        """
        logger.info(f"Using Ollama API at {self.base_url}")
        # No client to create as we'll use requests directly
        pass
    
    def _raw_model_call(self, model, chat_api_params):
        """
        Calls the Ollama API with the given parameters.
        """
        # Ollama API expects a different format compared to OpenAI
        ollama_params = {
            "model": model,
            "messages": chat_api_params.get("messages", []),
            "stream": False,
        }
        
        # Optionally add temperature if specified
        if "temperature" in chat_api_params and chat_api_params["temperature"] is not None:
            ollama_params["temperature"] = chat_api_params["temperature"]
            
        # Map max_tokens to num_predict for Ollama
        if "max_tokens" in chat_api_params and chat_api_params["max_tokens"] is not None:
            ollama_params["num_predict"] = chat_api_params["max_tokens"]
            
        # Make the API call
        logger.debug(f"Sending request to Ollama API: {ollama_params}")
        response = requests.post(f"{self.base_url}/api/chat", json=ollama_params)
        
        if response.status_code != 200:
            logger.error(f"Error from Ollama API: {response.status_code} {response.text}")
            raise Exception(f"Ollama API error: {response.status_code} {response.text}")
            
        # Convert Ollama response to OpenAI-like format
        result = response.json()
        
        # Create a structure similar to OpenAI's response
        openai_like_response = type('OllamaResponse', (), {
            'choices': [
                type('Choice', (), {
                    'message': type('Message', (), {
                        'to_dict': lambda: {"role": "assistant", "content": result.get("message", {}).get("content", "")}
                    })
                })
            ]
        })
        
        return openai_like_response
        
    def _raw_embedding_model_call(self, text, model):
        """
        Gets embeddings using Ollama API.
        """
        params = {
            "model": model,
            "prompt": text
        }
        
        response = requests.post(f"{self.base_url}/api/embeddings", json=params)
        
        if response.status_code != 200:
            logger.error(f"Error from Ollama API: {response.status_code} {response.text}")
            raise Exception(f"Ollama API error: {response.status_code} {response.text}")
            
        result = response.json()
        
        # Create a structure similar to OpenAI's response
        openai_like_response = type('OllamaEmbeddingResponse', (), {
            'data': [
                type('EmbeddingData', (), {
                    'embedding': result.get("embedding", [])
                })
            ]
        })
        
        return openai_like_response
    

###########################################################################
# Exceptions
###########################################################################
class InvalidRequestError(Exception):
    """
    Exception raised when the request to the OpenAI API is invalid.
    """
    pass

class NonTerminalError(Exception):
    """
    Exception raised when an unspecified error occurs but we know we can retry.
    """
    pass

###########################################################################
# Clients registry
#
# We can have potentially different clients, so we need a place to 
# register them and retrieve them when needed.
#
# We support both OpenAI and Azure OpenAI Service API by default.
# Thus, we need to set the API parameters based on the choice of the user.
# This is done within specialized classes.
#
# It is also possible to register custom clients, to access internal or
# otherwise non-conventional API endpoints.
###########################################################################
_api_type_to_client = {}
_api_type_override = None

def register_client(api_type, client):
    """
    Registers a client for the given API type.

    Args:
    api_type (str): The API type for which we want to register the client.
    client: The client to register.
    """
    _api_type_to_client[api_type] = client

def _get_client_for_api_type(api_type):
    """
    Returns the client for the given API type.

    Args:
    api_type (str): The API type for which we want to get the client.
    """
    try:
        return _api_type_to_client[api_type]
    except KeyError:
        raise ValueError(f"API type {api_type} is not supported. Please check the 'config.ini' file.")

def client():
    """
    Returns the client for the configured API type.
    """
    api_type = config["OpenAI"]["API_TYPE"] if _api_type_override is None else _api_type_override
    
    logger.debug(f"Using  API type {api_type}.")
    return _get_client_for_api_type(api_type)


# TODO simplify the custom configuration methods below

def force_api_type(api_type):
    """
    Forces the use of the given API type, thus overriding any other configuration.

    Args:
    api_type (str): The API type to use.
    """
    global _api_type_override
    _api_type_override = api_type

def force_api_cache(cache_api_calls, cache_file_name=default["cache_file_name"]):
    """
    Forces the use of the given API cache configuration, thus overriding any other configuration.

    Args:
    cache_api_calls (bool): Whether to cache API calls.
    cache_file_name (str): The name of the file to use for caching API calls.
    """
    # set the cache parameters on all clients
    for client in _api_type_to_client.values():
        client.set_api_cache(cache_api_calls, cache_file_name)

# default clients
register_client("openai", OpenAIClient())
register_client("azure", AzureClient())
register_client("ollama", OllamaClient())



