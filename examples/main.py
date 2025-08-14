import warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*")

from agentic_memory.memory_system import AgenticMemorySystem

import os

# Proxy settings
os.environ["no_proxy"]="*"
os.environ["NO_PROXY"]="*"

# HuggingFace offline mode settings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("Initializing Agentic Memory System...\n")
memory_system = AgenticMemorySystem(
    model_name='nomic-embed-text',   # Embedding model name 
    llm_backend="ollama",            # LLM backend (openai/ollama)
    llm_model="qwen3:8b",            # LLM model name
    embedding_backend="ollama"       # Use Ollama for embeddings (fully local)
)

# Add Memories with Automatic LLM Analysis
# Simple addition - LLM automatically generates keywords, context, and tags
print("Adding memory with automatic metadata generation...\n")
memory_id1 = memory_system.add_note(
    "Machine learning algorithms use neural networks to process complex datasets and identify patterns."
)

# Check the automatically generated metadata
print("Memory added with auto-generated metadata:\n")
memory = memory_system.read(memory_id1)
print(f"Content: {memory.content}")
print(f"Auto-generated Keywords: {memory.keywords}")  # e.g., ['machine learning', 'neural networks', 'datasets']
print(f"Auto-generated Context: {memory.context}")    # e.g., "Discussion about ML algorithms and data processing"
print(f"Auto-generated Tags: {memory.tags}")          # e.g., ['artificial intelligence', 'data science', 'technology']

# Partial metadata provision - LLM fills in missing attributes
memory_id2 = memory_system.add_note(
    content="Python is excellent for data science applications",
    keywords=["Python", "programming"]  # Provide keywords, LLM will generate context and tags
)

# Manual metadata provision - no LLM analysis needed
memory_id3 = memory_system.add_note(
    content="Project meeting notes for Q1 review",
    keywords=["meeting", "project", "review"],
    context="Business project management discussion",
    tags=["business", "project", "meeting"],
    timestamp="202503021500"  # YYYYMMDDHHmm format
)

# Enhanced Retrieval with Metadata
# The system now uses generated metadata for better semantic search
results = memory_system.search("artificial intelligence data processing", k=3)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Keywords: {result.get('keywords', [])}")
    print(f"Tags: {result.get('tags', [])}")
    print(f"Relevance Score: {result.get('score', 'N/A')}")
    print("---")

# Alternative search methods
results = memory_system.search_agentic("neural networks", k=5)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Tags: {result.get('tags', [])}")
    print("---")

# Update Memories
memory_system.update(memory_id1, content="Updated: Deep learning neural networks for pattern recognition")

# Delete Memories
memory_system.delete(memory_id3)

# Memory Evolution
# The system automatically evolves memories by:
# 1. Using LLM to analyze content and generate semantic metadata
# 2. Finding relationships using enhanced ChromaDB embeddings (content + metadata)
# 3. Updating tags, context, and connections based on related memories
# 4. Creating semantic links between memories
# This happens automatically when adding or updating memories!