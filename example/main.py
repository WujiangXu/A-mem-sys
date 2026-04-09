import os
from agentic_memory.memory_system import AgenticMemorySystem

AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT_NAME    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")


memory_system = AgenticMemorySystem(
    llm_backend="azure_openai",
    llm_model=AZURE_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    # Embedding backend
    embedding_provider="azure_openai",
    azure_embedding_model=AZURE_EMBEDDING_DEPLOYMENT,
    azure_embedding_api_key=AZURE_OPENAI_API_KEY,       
    azure_embedding_endpoint=AZURE_OPENAI_ENDPOINT,      
    azure_embedding_api_version=AZURE_OPENAI_API_VERSION,
    # Memory evolution
    evo_threshold=100,
)

# Add Memories with Automatic LLM Analysis ✨
# Simple addition - LLM automatically generates keywords, context, and tags
# memory_id1 = memory_system.add_note(
#     "Machine learning algorithms use neural networks to process complex datasets and identify patterns."
# )

# Check the automatically generated metadata
# memory = memory_system.read(memory_id1)
# print(f"Content: {memory.content}")
# print(f"Auto-generated Keywords: {memory.keywords}")  # e.g., ['machine learning', 'neural networks', 'datasets']
# print(f"Auto-generated Context: {memory.context}")    # e.g., "Discussion about ML algorithms and data processing"
# print(f"Auto-generated Tags: {memory.tags}")          # e.g., ['artificial intelligence', 'data science', 'technology']

# Partial metadata provision - LLM fills in missing attributes
memory_id2 = memory_system.add_note(
    content="Python is excellent for data science applications",
    keywords=["Python", "programming"]  # Provide keywords, LLM will generate context and tags
)

# Manual metadata provision - no LLM analysis needed
# memory_id3 = memory_system.add_note(
#     content="Project meeting notes for Q1 review",
#     keywords=["meeting", "project", "review"],
#     context="Business project management discussion",
#     tags=["business", "project", "meeting"],
#     timestamp="202503021500"  # YYYYMMDDHHmm format
# )

# Update Memories 🔄
memory_system.update(memory_id2, content="Updated: Deep learning neural networks for pattern recognition")

# Enhanced Retrieval with Metadata 🔍
# The system now uses generated metadata for better semantic search
results = memory_system.search("Programming languages the best when developing AI applications", k=3)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Keywords: {result['keywords']}")
    print(f"Tags: {result['tags']}")
    print(f"Relevance Score: {result.get('score', 'N/A')}")
    print("---")

# Alternative search methods
results = memory_system.search_agentic("Programming languages", k=5)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Tags: {result['tags']}")
    print("---")

# Delete Memories ❌
# memory_system.delete(memory_id3)