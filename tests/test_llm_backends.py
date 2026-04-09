import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
from agentic_memory.llm_controller import (
    LLMController,
    OpenAIController,
    OllamaController,
    SGLangController,
    AzureOpenAIController,
    OpenRouterController,
)


class TestSGLangController(unittest.TestCase):
    """Test SGLang backend controller"""

    def setUp(self):
        """Set up test environment before each test."""
        self.controller = SGLangController(
            model="meta-llama/Llama-3.1-8B-Instruct",
            sglang_host="http://localhost",
            sglang_port=30000
        )

    def test_initialization(self):
        """Test SGLangController initialization"""
        self.assertEqual(self.controller.model, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(self.controller.sglang_host, "http://localhost")
        self.assertEqual(self.controller.sglang_port, 30000)
        self.assertEqual(self.controller.base_url, "http://localhost:30000")

    def test_generate_empty_value(self):
        """Test _generate_empty_value helper method"""
        self.assertEqual(self.controller._generate_empty_value("array"), [])
        self.assertEqual(self.controller._generate_empty_value("string"), "")
        self.assertEqual(self.controller._generate_empty_value("object"), {})
        self.assertEqual(self.controller._generate_empty_value("number"), 0)
        self.assertEqual(self.controller._generate_empty_value("integer"), 0)
        self.assertEqual(self.controller._generate_empty_value("boolean"), False)
        self.assertIsNone(self.controller._generate_empty_value("unknown"))

    def test_generate_empty_response(self):
        """Test _generate_empty_response helper method"""
        response_format = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "keywords": {"type": "array"},
                        "context": {"type": "string"},
                        "tags": {"type": "array"}
                    }
                }
            }
        }

        result = self.controller._generate_empty_response(response_format)
        self.assertEqual(result["keywords"], [])
        self.assertEqual(result["context"], "")
        self.assertEqual(result["tags"], [])

    @patch('agentic_memory.llm_controller.requests.post')
    def test_get_completion_success(self, mock_post):
        """Test successful completion from SGLang server"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": '{"keywords": ["test"], "context": "Test context", "tags": ["test"]}'
        }
        mock_post.return_value = mock_response

        response_format = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "keywords": {"type": "array"},
                        "context": {"type": "string"},
                        "tags": {"type": "array"}
                    }
                }
            }
        }

        result = self.controller.get_completion(
            prompt="Test prompt",
            response_format=response_format,
            temperature=0.7
        )

        # Verify the request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Check URL
        self.assertEqual(call_args[0][0], "http://localhost:30000/generate")

        # Check payload
        payload = call_args[1]['json']
        self.assertEqual(payload['text'], "Test prompt")
        self.assertEqual(payload['sampling_params']['temperature'], 0.7)
        self.assertEqual(payload['sampling_params']['max_new_tokens'], 1000)

        # Check result
        self.assertIsNotNone(result)

    @patch('agentic_memory.llm_controller.requests.post')
    def test_get_completion_server_error(self, mock_post):
        """Test handling of SGLang server error"""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        response_format = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "keywords": {"type": "array"},
                        "context": {"type": "string"}
                    }
                }
            }
        }

        result = self.controller.get_completion(
            prompt="Test prompt",
            response_format=response_format
        )

        # Should return empty response on error
        result_dict = json.loads(result)
        self.assertEqual(result_dict["keywords"], [])
        self.assertEqual(result_dict["context"], "")

    @patch('agentic_memory.llm_controller.requests.post')
    def test_get_completion_network_error(self, mock_post):
        """Test handling of network error"""
        # Mock network error
        mock_post.side_effect = Exception("Connection refused")

        response_format = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "keywords": {"type": "array"}
                    }
                }
            }
        }

        result = self.controller.get_completion(
            prompt="Test prompt",
            response_format=response_format
        )

        # Should return empty response on error
        result_dict = json.loads(result)
        self.assertEqual(result_dict["keywords"], [])


class TestLLMControllerBackends(unittest.TestCase):
    """Test LLMController with different backends"""

    def test_openai_backend_initialization(self):
        """Test initialization with OpenAI backend"""
        with patch.object(OpenAIController, '__init__', return_value=None):
            controller = LLMController(
                backend="openai",
                model="gpt-4o-mini",
                api_key="test-key"
            )
            self.assertIsInstance(controller.llm, OpenAIController)

    # def test_ollama_backend_initialization(self):
    #     """Test initialization with Ollama backend"""
    #     with patch('agentic_memory.llm_controller.completion'):
    #         controller = LLMController(
    #             backend="ollama",
    #             model="llama2"
    #         )
    #         self.assertIsInstance(controller.llm, OllamaController)

    def test_sglang_backend_initialization(self):
        """Test initialization with SGLang backend"""
        controller = LLMController(
            backend="sglang",
            model="meta-llama/Llama-3.1-8B-Instruct",
            sglang_host="http://localhost",
            sglang_port=30000
        )
        self.assertIsInstance(controller.llm, SGLangController)
        self.assertEqual(controller.llm.model, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(controller.llm.base_url, "http://localhost:30000")

    def test_azure_openai_backend_initialization(self):
        """Test initialization with Azure OpenAI backend"""
        with patch.object(AzureOpenAIController, '__init__', return_value=None):
            controller = LLMController(
                backend="azure_openai",
                model="gpt-4o-mini",
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com/",
                api_version="2024-02-15-preview",
            )
            self.assertIsInstance(controller.llm, AzureOpenAIController)

    def test_openrouter_backend_initialization(self):
        """Test initialization with OpenRouter backend"""
        with patch.object(OpenRouterController, '__init__', return_value=None):
            controller = LLMController(
                backend="openrouter",
                model="openai/gpt-4o-mini",
                api_key="test-key",
            )
            self.assertIsInstance(controller.llm, OpenRouterController)

    def test_invalid_backend(self):
        """Test initialization with invalid backend raises error"""
        with self.assertRaises(ValueError) as context:
            LLMController(backend="invalid_backend")

        self.assertIn("Backend must be one of", str(context.exception))
        self.assertIn("azure_openai", str(context.exception))

    def test_sglang_custom_port(self):
        """Test SGLang with custom host and port"""
        controller = LLMController(
            backend="sglang",
            model="llama2",
            sglang_host="http://192.168.1.100",
            sglang_port=8080
        )
        self.assertEqual(controller.llm.base_url, "http://192.168.1.100:8080")


class TestAgenticMemorySystemWithSGLang(unittest.TestCase):
    """Test AgenticMemorySystem with SGLang backend"""

    @patch('agentic_memory.llm_controller.requests.post')
    def test_memory_system_with_sglang(self, mock_post):
        """Test creating AgenticMemorySystem with SGLang backend"""
        from agentic_memory.memory_system import AgenticMemorySystem

        # Mock SGLang responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": '{"keywords": ["test", "memory"], "context": "Testing SGLang", "tags": ["test"]}'
        }
        mock_post.return_value = mock_response

        # Create memory system with SGLang
        memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend="sglang",
            llm_model="meta-llama/Llama-3.1-8B-Instruct",
            sglang_host="http://localhost",
            sglang_port=30000
        )

        # Verify SGLang backend is used
        self.assertIsInstance(memory_system.llm_controller.llm, SGLangController)

    def test_sglang_parameters_passed_correctly(self):
        """Test that SGLang parameters are passed correctly to controller"""
        from agentic_memory.memory_system import AgenticMemorySystem

        memory_system = AgenticMemorySystem(
            llm_backend="sglang",
            llm_model="llama2",
            sglang_host="http://10.0.0.1",
            sglang_port=9999
        )

        self.assertEqual(memory_system.llm_controller.llm.sglang_host, "http://10.0.0.1")
        self.assertEqual(memory_system.llm_controller.llm.sglang_port, 9999)
        self.assertEqual(memory_system.llm_controller.llm.base_url, "http://10.0.0.1:9999")


class TestSGLangJSONSchemaFormat(unittest.TestCase):
    """Test JSON schema formatting for SGLang"""

    def setUp(self):
        self.controller = SGLangController()

    @patch('agentic_memory.llm_controller.requests.post')
    def test_json_schema_converted_to_string(self, mock_post):
        """Test that JSON schema is converted to string for SGLang"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "{}"}
        mock_post.return_value = mock_response

        schema = {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}}
            }
        }

        response_format = {"json_schema": {"schema": schema}}

        self.controller.get_completion(
            prompt="Test",
            response_format=response_format
        )

        # Verify json_schema was converted to string in payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']

        # json_schema should be a string in sampling_params
        self.assertIsInstance(payload['sampling_params']['json_schema'], str)

        # It should be the stringified version of the original schema
        parsed_schema = json.loads(payload['sampling_params']['json_schema'])
        self.assertEqual(parsed_schema, schema)


class TestAzureOpenAIController(unittest.TestCase):
    """Tests for AzureOpenAIController."""

    _ENDPOINT = "https://test-resource.openai.azure.com/"
    _API_KEY = "test-azure-key"
    _API_VERSION = "2024-02-15-preview"
    _MODEL = "gpt-4o-mini"

    def _make_controller(self, **kwargs):
        """Helper: create controller with minimal valid config."""
        defaults = dict(
            model=self._MODEL,
            api_key=self._API_KEY,
            azure_endpoint=self._ENDPOINT,
            api_version=self._API_VERSION,
        )
        defaults.update(kwargs)
        with patch("agentic_memory.llm_controller.AzureOpenAIController.__init__",
                   wraps=AzureOpenAIController.__init__):
            # Patch actual AzureOpenAI client so no real HTTP call happens
            with patch("openai.AzureOpenAI"):
                return AzureOpenAIController(**defaults)

    def test_initialization(self):
        """Test AzureOpenAIController stores model and creates client."""
        with patch("openai.AzureOpenAI") as mock_client_cls:
            ctrl = AzureOpenAIController(
                model=self._MODEL,
                api_key=self._API_KEY,
                azure_endpoint=self._ENDPOINT,
                api_version=self._API_VERSION,
            )
        self.assertEqual(ctrl.model, self._MODEL)
        mock_client_cls.assert_called_once_with(
            api_key=self._API_KEY,
            azure_endpoint=self._ENDPOINT,
            api_version=self._API_VERSION,
        )

    def test_missing_api_key_raises_error(self):
        """ValueError raised when API key is absent."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AZURE_OPENAI_API_KEY", None)
            with self.assertRaises(ValueError, msg="Should raise ValueError for missing API key"):
                AzureOpenAIController(
                    model=self._MODEL,
                    api_key=None,
                    azure_endpoint=self._ENDPOINT,
                    api_version=self._API_VERSION,
                )

    def test_missing_endpoint_raises_error(self):
        """ValueError raised when endpoint is absent."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
            with self.assertRaises(ValueError, msg="Should raise ValueError for missing endpoint"):
                AzureOpenAIController(
                    model=self._MODEL,
                    api_key=self._API_KEY,
                    azure_endpoint=None,
                    api_version=self._API_VERSION,
                )

    def test_env_var_fallback(self):
        """Controller picks up credentials from environment variables."""
        env = {
            "AZURE_OPENAI_API_KEY": self._API_KEY,
            "AZURE_OPENAI_ENDPOINT": self._ENDPOINT,
            "AZURE_OPENAI_API_VERSION": self._API_VERSION,
        }
        with patch.dict(os.environ, env):
            with patch("openai.AzureOpenAI") as mock_client_cls:
                ctrl = AzureOpenAIController(model=self._MODEL)
        mock_client_cls.assert_called_once_with(
            api_key=self._API_KEY,
            azure_endpoint=self._ENDPOINT,
            api_version=self._API_VERSION,
        )

    def test_get_completion_success(self):
        """get_completion returns the message content on success."""
        mock_message = Mock()
        mock_message.content = '{"keywords": ["azure"], "context": "cloud", "tags": ["ai"]}'
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch("openai.AzureOpenAI") as mock_client_cls:
            mock_client_cls.return_value.chat.completions.create.return_value = mock_response
            ctrl = AzureOpenAIController(
                model=self._MODEL,
                api_key=self._API_KEY,
                azure_endpoint=self._ENDPOINT,
                api_version=self._API_VERSION,
            )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "context": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        }

        result = ctrl.get_completion(
            prompt="What are the keywords?",
            response_format=response_format,
            temperature=0.7,
        )

        parsed = json.loads(result)
        self.assertEqual(parsed["keywords"], ["azure"])
        self.assertEqual(parsed["context"], "cloud")
        self.assertEqual(parsed["tags"], ["ai"])

        # Verify model, messages, and temperature were forwarded
        create_call = ctrl.client.chat.completions.create.call_args
        kwargs = create_call[1] if create_call[1] else create_call[0][0]
        self.assertEqual(kwargs["model"], self._MODEL)
        self.assertAlmostEqual(kwargs["temperature"], 0.7)

    def test_get_completion_with_max_tokens(self):
        """max_tokens kwarg is forwarded to the API call."""
        mock_message = Mock()
        mock_message.content = '{"keywords": []}'
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch("openai.AzureOpenAI") as mock_client_cls:
            mock_client_cls.return_value.chat.completions.create.return_value = mock_response
            ctrl = AzureOpenAIController(
                model=self._MODEL,
                api_key=self._API_KEY,
                azure_endpoint=self._ENDPOINT,
                api_version=self._API_VERSION,
            )

        response_format = {"type": "json_object"}
        ctrl.get_completion(prompt="test", response_format=response_format, max_tokens=512)

        create_call = ctrl.client.chat.completions.create.call_args
        kwargs = create_call[1] if create_call[1] else create_call[0][0]
        self.assertEqual(kwargs["max_tokens"], 512)


class TestAzureOpenAIEmbeddingFunction(unittest.TestCase):
    """Tests for AzureOpenAIEmbeddingFunction in retrievers."""

    _ENDPOINT = "https://test-resource.openai.azure.com/"
    _API_KEY = "test-embed-key"
    _MODEL = "text-embedding-3-small"

    def _make_ef(self, **kwargs):
        defaults = dict(
            model=self._MODEL,
            api_key=self._API_KEY,
            azure_endpoint=self._ENDPOINT,
            api_version="2024-02-15-preview",
        )
        defaults.update(kwargs)
        with patch("openai.AzureOpenAI"):
            from agentic_memory.retrievers import AzureOpenAIEmbeddingFunction
            return AzureOpenAIEmbeddingFunction(**defaults)

    def test_name(self):
        """name() returns the expected identifier."""
        ef = self._make_ef()
        self.assertEqual(ef.name(), "azure_openai_embedding")

    def test_missing_api_key_raises(self):
        """ValueError when API key is missing."""
        from agentic_memory.retrievers import AzureOpenAIEmbeddingFunction
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AZURE_OPENAI_API_KEY", None)
            with self.assertRaises(ValueError):
                AzureOpenAIEmbeddingFunction(
                    model=self._MODEL,
                    api_key=None,
                    azure_endpoint=self._ENDPOINT,
                )

    def test_missing_endpoint_raises(self):
        """ValueError when endpoint is missing."""
        from agentic_memory.retrievers import AzureOpenAIEmbeddingFunction
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
            with self.assertRaises(ValueError):
                AzureOpenAIEmbeddingFunction(
                    model=self._MODEL,
                    api_key=self._API_KEY,
                    azure_endpoint=None,
                )

    def test_call_returns_embeddings(self):
        """__call__ returns a list of embedding vectors."""
        fake_embedding = [0.1] * 1536
        mock_item = Mock()
        mock_item.embedding = fake_embedding
        mock_response = Mock()
        mock_response.data = [mock_item, mock_item]

        with patch("openai.AzureOpenAI") as mock_cls:
            mock_cls.return_value.embeddings.create.return_value = mock_response
            from agentic_memory.retrievers import AzureOpenAIEmbeddingFunction
            ef = AzureOpenAIEmbeddingFunction(
                model=self._MODEL,
                api_key=self._API_KEY,
                azure_endpoint=self._ENDPOINT,
            )

        result = ef(["hello world", "azure openai"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], fake_embedding)
        ef.client.embeddings.create.assert_called_once_with(
            model=self._MODEL, input=["hello world", "azure openai"]
        )

    def test_embed_query_accepts_string(self):
        """embed_query wraps a plain string in a list."""
        fake_embedding = [0.0] * 1536
        mock_item = Mock()
        mock_item.embedding = fake_embedding
        mock_response = Mock()
        mock_response.data = [mock_item]

        with patch("openai.AzureOpenAI") as mock_cls:
            mock_cls.return_value.embeddings.create.return_value = mock_response
            from agentic_memory.retrievers import AzureOpenAIEmbeddingFunction
            ef = AzureOpenAIEmbeddingFunction(
                model=self._MODEL,
                api_key=self._API_KEY,
                azure_endpoint=self._ENDPOINT,
            )

        result = ef.embed_query("single string query")
        self.assertEqual(len(result), 1)
        ef.client.embeddings.create.assert_called_once_with(
            model=self._MODEL, input=["single string query"]
        )

    def test_embed_documents(self):
        """embed_documents forwards list to the API."""
        fake_embedding = [0.5] * 1536
        mock_item = Mock()
        mock_item.embedding = fake_embedding
        mock_response = Mock()
        mock_response.data = [mock_item]

        with patch("openai.AzureOpenAI") as mock_cls:
            mock_cls.return_value.embeddings.create.return_value = mock_response
            from agentic_memory.retrievers import AzureOpenAIEmbeddingFunction
            ef = AzureOpenAIEmbeddingFunction(
                model=self._MODEL,
                api_key=self._API_KEY,
                azure_endpoint=self._ENDPOINT,
            )

        result = ef.embed_documents(["doc text"])
        self.assertEqual(result[0], fake_embedding)


class TestOpenRouterController(unittest.TestCase):
    """Tests for OpenRouterController."""

    _API_KEY = "test-openrouter-key"

    def test_initialization_adds_prefix(self):
        """Model without 'openrouter/' prefix gets it prepended automatically."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": self._API_KEY}):
            ctrl = OpenRouterController(model="openai/gpt-4o-mini")
        self.assertEqual(ctrl.model, "openrouter/openai/gpt-4o-mini")

    def test_initialization_keeps_existing_prefix(self):
        """Model already prefixed with 'openrouter/' is not double-prefixed."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": self._API_KEY}):
            ctrl = OpenRouterController(model="openrouter/openai/gpt-4o-mini")
        self.assertEqual(ctrl.model, "openrouter/openai/gpt-4o-mini")

    def test_initialization_explicit_api_key(self):
        """Explicit api_key takes precedence over environment variable."""
        ctrl = OpenRouterController(model="openai/gpt-4o-mini", api_key=self._API_KEY)
        self.assertEqual(ctrl.api_key, self._API_KEY)
        self.assertEqual(os.environ.get("OPENROUTER_API_KEY"), self._API_KEY)

    def test_missing_api_key_raises_error(self):
        """ValueError is raised when no API key is provided or set in environment."""
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                OpenRouterController(model="openai/gpt-4o-mini", api_key=None)

    @patch("agentic_memory.llm_controller.completion")
    def test_get_completion_success(self, mock_completion):
        """get_completion returns message content from litellm on success."""
        mock_completion.return_value.choices[0].message.content = (
            '{"keywords": ["openrouter", "test"], "context": "Testing", "tags": ["llm"]}'
        )
        response_format = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "keywords": {"type": "array"},
                        "context": {"type": "string"},
                        "tags": {"type": "array"},
                    }
                }
            }
        }

        ctrl = OpenRouterController(model="openai/gpt-4o-mini", api_key=self._API_KEY)
        result = ctrl.get_completion("Test prompt", response_format, temperature=0.5)

        mock_completion.assert_called_once_with(
            model="openrouter/openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": "Test prompt"},
            ],
            response_format=response_format,
            temperature=0.5,
        )
        result_dict = json.loads(result)
        self.assertEqual(result_dict["keywords"], ["openrouter", "test"])

    @patch("agentic_memory.llm_controller.completion", side_effect=Exception("API error"))
    def test_get_completion_fallback_on_error(self, mock_completion):
        """get_completion returns empty schema-matching response when litellm raises."""
        response_format = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "keywords": {"type": "array"},
                        "context": {"type": "string"},
                    }
                }
            }
        }

        ctrl = OpenRouterController(model="openai/gpt-4o-mini", api_key=self._API_KEY)
        result = ctrl.get_completion("Test prompt", response_format)

        result_dict = json.loads(result)
        self.assertEqual(result_dict["keywords"], [])
        self.assertEqual(result_dict["context"], "")

    @patch("agentic_memory.llm_controller.completion")
    def test_get_completion_default_temperature(self, mock_completion):
        """get_completion uses temperature=1.0 by default."""
        mock_completion.return_value.choices[0].message.content = "{}"
        response_format = {"json_schema": {"schema": {"properties": {}}}}

        ctrl = OpenRouterController(model="openai/gpt-4o-mini", api_key=self._API_KEY)
        ctrl.get_completion("prompt", response_format)

        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 1.0)


class TestAgenticMemorySystemWithAzureBackend(unittest.TestCase):
    """Test AgenticMemorySystem initialised with Azure OpenAI (LLM + embeddings)."""

    _CREDS = dict(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/",
        api_version="2024-02-15-preview",
        azure_embedding_api_key="test-embed-key",
        azure_embedding_endpoint="https://test.openai.azure.com/",
        azure_embedding_api_version="2024-02-15-preview",
    )

    @patch("openai.AzureOpenAI")
    def test_azure_llm_backend_used(self, _mock_azure):
        """AgenticMemorySystem selects AzureOpenAIController for llm_backend='azure_openai'."""
        from agentic_memory.memory_system import AgenticMemorySystem

        ms = AgenticMemorySystem(
            llm_backend="azure_openai",
            llm_model="gpt-4o-mini",
            **self._CREDS,
        )
        self.assertIsInstance(ms.llm_controller.llm, AzureOpenAIController)

    @patch("openai.AzureOpenAI")
    def test_azure_embedding_provider_used(self, _mock_azure):
        """ChromaRetriever uses AzureOpenAIEmbeddingFunction when embedding_provider='azure_openai'."""
        from agentic_memory.memory_system import AgenticMemorySystem
        from agentic_memory.retrievers import AzureOpenAIEmbeddingFunction

        ms = AgenticMemorySystem(
            llm_backend="azure_openai",
            llm_model="gpt-4o-mini",
            embedding_provider="azure_openai",
            azure_embedding_model="text-embedding-3-small",
            **self._CREDS,
        )
        self.assertIsInstance(ms.retriever.embedding_function, AzureOpenAIEmbeddingFunction)

    @patch("openai.AzureOpenAI")
    def test_azure_embedding_model_stored(self, _mock_azure):
        """The embedding deployment name is forwarded to the embedding function."""
        from agentic_memory.memory_system import AgenticMemorySystem

        ms = AgenticMemorySystem(
            llm_backend="azure_openai",
            llm_model="gpt-4o-mini",
            embedding_provider="azure_openai",
            azure_embedding_model="text-embedding-3-small",
            **self._CREDS,
        )
        self.assertEqual(ms.retriever.embedding_function.model, "text-embedding-3-small")

    @patch("openai.AzureOpenAI")
    def test_sentence_transformer_still_default(self, _mock_azure):
        """Default embedding_provider remains sentence_transformer (no breaking change)."""
        from agentic_memory.memory_system import AgenticMemorySystem
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        ms = AgenticMemorySystem(
            llm_backend="azure_openai",
            llm_model="gpt-4o-mini",
            # embedding_provider NOT set → should default to sentence_transformer
            **self._CREDS,
        )
        self.assertIsInstance(ms.retriever.embedding_function, SentenceTransformerEmbeddingFunction)


if __name__ == '__main__':
    unittest.main()
