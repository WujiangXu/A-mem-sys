import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
from datetime import datetime

class TestAgenticMemorySystem(unittest.TestCase):
    # LLM response mocks
    _OPENAI_LLM_RESPONSE = json.dumps({
        "keywords": ["test", "memory"],
        "context": "Unit test context",
        "tags": ["test", "unit"],
    })
    _AZURE_LLM_RESPONSE = json.dumps({
        "keywords": ["azure", "openai", "memory"],
        "context": "Azure OpenAI integration test",
        "tags": ["cloud", "ai", "testing"],
    })

    # ------------------------------------------------------------------
    # Mock helpers
    # ------------------------------------------------------------------
    def _make_openai_completion_mock(self):
        msg = Mock()
        msg.content = self._OPENAI_LLM_RESPONSE
        choice = Mock()
        choice.message = msg
        response = Mock()
        response.choices = [choice]
        return response

    def _make_azure_completion_mock(self):
        msg = Mock()
        msg.content = self._AZURE_LLM_RESPONSE
        choice = Mock()
        choice.message = msg
        response = Mock()
        response.choices = [choice]
        return response

    def _make_azure_embedding_mock(self):
        item = Mock()
        item.embedding = [0.1] * 384
        resp = Mock()
        resp.data = [item]
        return resp

    def _make_azure_memory_system(self, mock_azure_cls):
        """Create an AgenticMemorySystem backed by a mocked Azure OpenAI client."""
        import chromadb
        from chromadb.config import Settings
        # Clear the shared in-memory store so the Azure EF can own the "memories" collection
        try:
            chromadb.Client(Settings(allow_reset=True)).reset()
        except Exception:
            pass
        instance = mock_azure_cls.return_value
        instance.chat.completions.create.return_value = self._make_azure_completion_mock()
        instance.embeddings.create.return_value = self._make_azure_embedding_mock()
        return AgenticMemorySystem(
            llm_backend="azure_openai",
            llm_model="gpt-4o-mini",
            api_key="test-llm-key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-02-15-preview",
            embedding_provider="azure_openai",
            azure_embedding_model="text-embedding-3-small",
            azure_embedding_api_key="test-embed-key",
            azure_embedding_endpoint="https://test.openai.azure.com/",
            azure_embedding_api_version="2024-02-15-preview",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def setUp(self):
        """Set up test environment before each test."""
        self._openai_patcher = patch("openai.OpenAI")
        mock_openai_cls = self._openai_patcher.start()
        mock_openai_cls.return_value.chat.completions.create.return_value = (
            self._make_openai_completion_mock()
        )
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai",
            llm_model="gpt-4o-mini",
            api_key="test-key",
        )

    def tearDown(self):
        self._openai_patcher.stop()
        # Reset shared ChromaDB in-memory store so the next test starts with a clean collection
        try:
            self.memory_system.retriever.client.reset()
        except Exception:
            pass
        
    def test_create_memory(self):
        """Test creating a new memory with complete metadata."""
        content = "Test memory content"
        tags = ["test", "memory"]
        keywords = ["test", "content"]
        links = ["link1", "link2"]
        context = "Test context"
        category = "Test category"
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        
        memory_id = self.memory_system.add_note(
            content=content,
            tags=tags,
            keywords=keywords,
            links=links,
            context=context,
            category=category,
            timestamp=timestamp
        )
        
        # Verify memory was created
        self.assertIsNotNone(memory_id)
        memory = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, content)
        self.assertEqual(memory.tags, tags)
        self.assertEqual(memory.keywords, keywords)
        self.assertEqual(memory.links, links)
        self.assertEqual(memory.context, context)
        self.assertEqual(memory.category, category)
        self.assertEqual(memory.timestamp, timestamp)
        
    def test_memory_metadata_persistence(self):
        """Test that memory metadata persists through ChromaDB storage and retrieval."""
        # Create a memory with complex metadata
        content = "Complex test memory"
        tags = ["test", "complex", "metadata"]
        keywords = ["test", "complex", "keywords"]
        links = ["link1", "link2", "link3"]
        context = "Complex test context"
        category = "Complex test category"
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        evolution_history = ["evolution1", "evolution2"]
        
        memory_id = self.memory_system.add_note(
            content=content,
            tags=tags,
            keywords=keywords,
            links=links,
            context=context,
            category=category,
            timestamp=timestamp,
            evolution_history=evolution_history
        )
        
        # Search for the memory using ChromaDB
        results = self.memory_system.search_agentic(content, k=1)
        self.assertGreater(len(results), 0)
        
        # Verify metadata in search results
        result = results[0]
        self.assertEqual(result['content'], content)
        self.assertEqual(result['tags'], tags)
        self.assertEqual(result['keywords'], keywords)
        self.assertEqual(result['context'], context)
        self.assertEqual(result['category'], category)
        
    def test_memory_update(self):
        """Test updating memory metadata through ChromaDB."""
        # Create initial memory
        content = "Initial content"
        memory_id = self.memory_system.add_note(content=content)
        
        # Update memory with new metadata
        new_content = "Updated content"
        new_tags = ["updated", "tags"]
        new_keywords = ["updated", "keywords"]
        new_context = "Updated context"
        
        success = self.memory_system.update(
            memory_id,
            content=new_content,
            tags=new_tags,
            keywords=new_keywords,
            context=new_context
        )
        
        self.assertTrue(success)
        
        # Verify updates in ChromaDB
        results = self.memory_system.search_agentic(new_content, k=1)
        self.assertGreater(len(results), 0)
        result = results[0]
        self.assertEqual(result['content'], new_content)
        self.assertEqual(result['tags'], new_tags)
        self.assertEqual(result['keywords'], new_keywords)
        self.assertEqual(result['context'], new_context)
        
    def test_memory_relationships(self):
        """Test memory relationships and linked memories."""
        # Create related memories
        content1 = "First memory"
        content2 = "Second memory"
        content3 = "Third memory"
        
        id1 = self.memory_system.add_note(content1)
        id2 = self.memory_system.add_note(content2)
        id3 = self.memory_system.add_note(content3)
        
        # Add relationships
        memory1 = self.memory_system.read(id1)
        memory2 = self.memory_system.read(id2)
        memory3 = self.memory_system.read(id3)
        
        memory1.links.append(id2)
        memory2.links.append(id1)
        memory2.links.append(id3)
        memory3.links.append(id2)
        
        # Update memories with relationships
        self.memory_system.update(id1, links=memory1.links)
        self.memory_system.update(id2, links=memory2.links)
        self.memory_system.update(id3, links=memory3.links)
        
        # Test relationship retrieval
        results = self.memory_system.search_agentic(content1, k=3)
        self.assertGreater(len(results), 0)
        
        # Verify relationships are maintained
        memory1_updated = self.memory_system.read(id1)
        self.assertIn(id2, memory1_updated.links)
        
    def test_memory_evolution(self):
        """Test memory evolution system with ChromaDB."""
        # Create related memories
        contents = [
            "Deep learning neural networks",
            "Neural network architectures",
            "Training deep neural networks"
        ]
        
        memory_ids = []
        for content in contents:
            memory_id = self.memory_system.add_note(content)
            memory_ids.append(memory_id)
            
        # Verify that memories have been properly evolved
        for memory_id in memory_ids:
            memory = self.memory_system.read(memory_id)
            self.assertIsNotNone(memory.tags)
            self.assertIsNotNone(memory.context)
            self.assertIsNotNone(memory.keywords)
            
        # Test evolution through search
        results = self.memory_system.search_agentic("neural networks", k=3)
        self.assertGreater(len(results), 0)
        
        # Verify evolution metadata
        for result in results:
            self.assertIsNotNone(result['tags'])
            self.assertIsNotNone(result['context'])
            self.assertIsNotNone(result['keywords'])
            
    def test_memory_deletion(self):
        """Test memory deletion from ChromaDB."""
        # Create and delete a memory
        content = "Memory to delete"
        memory_id = self.memory_system.add_note(content)
        
        # Verify memory exists
        memory = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory)
        
        # Delete memory
        success = self.memory_system.delete(memory_id)
        self.assertTrue(success)
        
        # Verify deletion
        memory = self.memory_system.read(memory_id)
        self.assertIsNone(memory)
        
        # Verify memory is removed from ChromaDB
        results = self.memory_system.search_agentic(content, k=1)
        self.assertEqual(len(results), 0)
        
    def test_memory_consolidation(self):
        """Test memory consolidation with ChromaDB."""
        # Use semantically distinct content so vector search reliably returns the right one
        contents = [
            "Python is a programming language used for data science",
            "Football is a popular sport played with a round ball",
            "Cooking pasta requires boiling water and adding salt",
        ]

        ids = []
        for content in contents:
            mem_id = self.memory_system.add_note(content)
            ids.append(mem_id)

        # Force consolidation
        self.memory_system.consolidate_memories()

        # Verify all memories are still accessible by ID
        for mem_id, content in zip(ids, contents):
            note = self.memory_system.read(mem_id)
            self.assertIsNotNone(note)
            self.assertEqual(note.content, content)
            
    def test_find_related_memories(self):
        """Test finding related memories."""
        # Create test memories
        contents = [
            "Python programming language",
            "Python data science",
            "Machine learning with Python",
            "Web development with JavaScript"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Test finding related memories
        results = self.memory_system.find_related_memories("Python", k=2)
        self.assertGreater(len(results), 0)
        
    def test_find_related_memories_raw(self):
        """Test finding related memories with raw format."""
        # Create test memories
        contents = [
            "Python programming language",
            "Python data science",
            "Machine learning with Python"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Test finding related memories in raw format
        results = self.memory_system.find_related_memories_raw("Python", k=2)
        self.assertIsNotNone(results)
        
    def test_process_memory(self):
        """Test memory processing and evolution."""
        # Create a test memory
        content = "Test memory for processing"
        memory_id = self.memory_system.add_note(content)
        
        # Get the memory
        memory = self.memory_system.read(memory_id)
        
        # Process the memory
        should_evolve, processed_memory = self.memory_system.process_memory(memory)
        
        # Verify processing results
        self.assertIsInstance(should_evolve, bool)
        self.assertIsInstance(processed_memory, MemoryNote)
        self.assertIsNotNone(processed_memory.tags)
        self.assertIsNotNone(processed_memory.context)
        self.assertIsNotNone(processed_memory.keywords)

    # ------------------------------------------------------------------
    # Azure OpenAI backend tests
    # ------------------------------------------------------------------
    @patch("openai.AzureOpenAI")
    def test_azure_llm_controller_is_azure(self, mock_azure_cls):
        """AgenticMemorySystem uses AzureOpenAIController for LLM calls."""
        from agentic_memory.llm_controller import AzureOpenAIController
        ms = self._make_azure_memory_system(mock_azure_cls)
        self.assertIsInstance(ms.llm_controller.llm, AzureOpenAIController)

    @patch("openai.AzureOpenAI")
    def test_azure_embedding_function_is_azure(self, mock_azure_cls):
        """ChromaRetriever uses AzureOpenAIEmbeddingFunction when embedding_provider='azure_openai'."""
        from agentic_memory.retrievers import AzureOpenAIEmbeddingFunction
        ms = self._make_azure_memory_system(mock_azure_cls)
        self.assertIsInstance(ms.retriever.embedding_function, AzureOpenAIEmbeddingFunction)

    @patch("openai.AzureOpenAI")
    def test_azure_add_note_returns_id(self, mock_azure_cls):
        """add_note returns a non-empty memory ID when using Azure backend."""
        ms = self._make_azure_memory_system(mock_azure_cls)
        memory_id = ms.add_note(
            content="Azure OpenAI provides GPT-4 models via a managed cloud service."
        )
        self.assertIsNotNone(memory_id)
        self.assertIsInstance(memory_id, str)
        self.assertGreater(len(memory_id), 0)

    @patch("openai.AzureOpenAI")
    def test_azure_add_note_stores_llm_metadata(self, mock_azure_cls):
        """Keywords/context/tags generated by Azure LLM are persisted on the MemoryNote."""
        ms = self._make_azure_memory_system(mock_azure_cls)
        mem_id = ms.add_note(content="Test cloud AI content")
        note = ms.read(mem_id)
        self.assertIsNotNone(note)
        self.assertIsInstance(note.keywords, list)
        self.assertIsInstance(note.tags, list)
        self.assertIsInstance(note.context, str)

    @patch("openai.AzureOpenAI")
    def test_azure_update_replaces_fields(self, mock_azure_cls):
        """update() replaces targeted fields on an Azure-backed memory."""
        ms = self._make_azure_memory_system(mock_azure_cls)
        mem_id = ms.add_note(content="Original content")
        success = ms.update(
            mem_id,
            content="Updated content",
            tags=["updated"],
            keywords=["updated"],
            context="Updated context",
        )
        self.assertTrue(success)
        note = ms.read(mem_id)
        self.assertEqual(note.content, "Updated content")
        self.assertEqual(note.tags, ["updated"])
        self.assertEqual(note.context, "Updated context")

    @patch("openai.AzureOpenAI")
    def test_azure_delete_removes_memory(self, mock_azure_cls):
        """delete() removes a memory from an Azure-backed system."""
        ms = self._make_azure_memory_system(mock_azure_cls)
        mem_id = ms.add_note(content="Memory to delete")
        self.assertIsNotNone(ms.read(mem_id))
        success = ms.delete(mem_id)
        self.assertTrue(success)
        self.assertIsNone(ms.read(mem_id))

    @patch("openai.AzureOpenAI")
    def test_azure_embedding_kwargs_propagated_to_consolidate(self, mock_azure_cls):
        """After consolidate_memories(), retriever still uses Azure embedding function."""
        from agentic_memory.retrievers import AzureOpenAIEmbeddingFunction
        ms = self._make_azure_memory_system(mock_azure_cls)
        ms.add_note(content="Memory before consolidation")
        ms.consolidate_memories()
        self.assertIsInstance(ms.retriever.embedding_function, AzureOpenAIEmbeddingFunction)


if __name__ == '__main__':
    unittest.main()
