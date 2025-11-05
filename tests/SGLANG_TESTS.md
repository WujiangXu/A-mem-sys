# SGLang Testing Guide

This document describes the SGLang backend tests in `test_llm_backends.py`.

## Test Classes

### 1. TestSGLangController
Tests the core SGLangController functionality.

**Tests:**
- `test_initialization` - Verifies correct initialization with host, port, and model
- `test_generate_empty_value` - Tests helper method for generating empty values by type
- `test_generate_empty_response` - Tests generation of empty JSON responses for error fallback
- `test_get_completion_success` - Tests successful HTTP communication with SGLang server
- `test_get_completion_server_error` - Tests graceful handling of 500 server errors
- `test_get_completion_network_error` - Tests graceful handling of network errors

### 2. TestLLMControllerBackends
Tests LLMController initialization with different backends.

**Tests:**
- `test_openai_backend_initialization` - Verifies OpenAI backend can be created
- `test_ollama_backend_initialization` - Verifies Ollama backend can be created
- `test_sglang_backend_initialization` - Verifies SGLang backend can be created with correct parameters
- `test_invalid_backend` - Ensures invalid backend names raise ValueError
- `test_sglang_custom_port` - Tests SGLang with custom host/port configuration

### 3. TestAgenticMemorySystemWithSGLang
Tests integration of SGLang with AgenticMemorySystem.

**Tests:**
- `test_memory_system_with_sglang` - Verifies AgenticMemorySystem can use SGLang backend
- `test_sglang_parameters_passed_correctly` - Ensures host/port parameters flow correctly

### 4. TestSGLangJSONSchemaFormat
Tests JSON schema formatting specific to SGLang.

**Tests:**
- `test_json_schema_converted_to_string` - Verifies JSON schema is stringified for SGLang API

## Running SGLang Tests

### Run all backend tests:
```bash
python -m unittest tests.test_llm_backends
```

### Run specific test class:
```bash
python -m unittest tests.test_llm_backends.TestSGLangController
```

### Run specific test:
```bash
python -m unittest tests.test_llm_backends.TestSGLangController.test_initialization
```

## Mock vs Integration Testing

**Mock Tests (No SGLang Server Required):**
- All tests in `test_llm_backends.py` use mocks and don't require a running SGLang server
- HTTP requests are mocked using `unittest.mock.patch`
- Suitable for CI/CD pipelines

**Integration Tests (Require SGLang Server):**
To test with a real SGLang server:

1. Start SGLang server:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```

2. Run integration test:
```python
from agentic_memory.memory_system import AgenticMemorySystem

memory_system = AgenticMemorySystem(
    llm_backend="sglang",
    llm_model="meta-llama/Llama-3.1-8B-Instruct",
    sglang_host="http://localhost",
    sglang_port=30000
)

# Add a memory - this will hit the real SGLang server
memory_id = memory_system.add_note("Test SGLang integration")
memory = memory_system.read(memory_id)
print(f"Keywords: {memory.keywords}")
print(f"Context: {memory.context}")
print(f"Tags: {memory.tags}")
```

## Test Coverage

The SGLang tests cover:
- ✅ Initialization and configuration
- ✅ HTTP request formatting (payload, headers, URL)
- ✅ JSON schema string conversion
- ✅ Successful response handling
- ✅ Error handling (server errors, network errors)
- ✅ Fallback to empty responses on failure
- ✅ Integration with LLMController factory
- ✅ Integration with AgenticMemorySystem
- ✅ Custom host/port configuration

## Key Differences from OpenAI/Ollama

**SGLang-Specific Behaviors:**
1. JSON schema must be stringified (not a dict object)
2. Uses HTTP POST to `/generate` endpoint
3. Schema passed in `sampling_params` not top-level
4. Response contains `text` field, not `choices` array
5. Configurable host/port for distributed deployment
