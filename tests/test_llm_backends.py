import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from agentic_memory.llm_controller import (
    LLMController,
    OpenAIController,
    OllamaController,
    SGLangController
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

    def test_ollama_backend_initialization(self):
        """Test initialization with Ollama backend"""
        with patch('agentic_memory.llm_controller.completion'):
            controller = LLMController(
                backend="ollama",
                model="llama2"
            )
            self.assertIsInstance(controller.llm, OllamaController)

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

    def test_invalid_backend(self):
        """Test initialization with invalid backend raises error"""
        with self.assertRaises(ValueError) as context:
            LLMController(backend="invalid_backend")

        self.assertIn("Backend must be one of", str(context.exception))

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


if __name__ == '__main__':
    unittest.main()
