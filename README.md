# Agentic Memory 🧠

A novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way.

## Introduction 🌟

Large Language Model (LLM) agents have demonstrated remarkable capabilities in handling complex real-world tasks through external tool usage. However, to effectively leverage historical experiences, they require sophisticated memory systems. Traditional memory systems, while providing basic storage and retrieval functionality, often lack advanced memory organization capabilities.

Our project introduces an innovative **Agentic Memory** system that revolutionizes how LLM agents manage and utilize their memories:

<div align="center">
  <img src="Figure/intro-a.jpg" alt="Traditional Memory System" width="600"/>
  <img src="Figure/intro-b.jpg" alt="Our Proposed Agentic Memory" width="600"/>
  <br>
  <em>Comparison between traditional memory system (top) and our proposed agentic memory (bottom). Our system enables dynamic memory operations and flexible agent-memory interactions.</em>
</div>

> **Note:** This repository provides a memory system to facilitate agent construction. If you want to reproduce the results presented in our paper, please refer to: [https://github.com/WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory)

For more details, please refer to our paper: [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110)


## Key Features ✨

- 🔄 Dynamic memory organization based on Zettelkasten principles
- 🔍 Intelligent indexing and linking of memories via ChromaDB
- 📝 Comprehensive note generation with structured attributes
- 🌐 Interconnected knowledge networks
- 🧬 Continuous memory evolution and refinement
- 🤖 Agent-driven decision making for adaptive memory management

## Framework 🏗️

<div align="center">
  <img src="Figure/framework.jpg" alt="Agentic Memory Framework" width="800"/>
  <br>
  <em>The framework of our Agentic Memory system showing the dynamic interaction between LLM agents and memory components.</em>
</div>

## How It Works 🛠️

When a new memory is added to the system:
1. **LLM Analysis**: Automatically analyzes content to generate keywords, context, and tags (if not provided)
2. **Enhanced Embedding**: Creates vector embeddings using both content and generated metadata for superior retrieval
3. **Semantic Storage**: Stores memories in ChromaDB with rich semantic information
4. **Relationship Analysis**: Analyzes historical memories for relevant connections using enhanced embeddings
5. **Dynamic Linking**: Establishes meaningful links based on content and metadata similarities
6. **Memory Evolution**: Enables continuous memory evolution and updates through intelligent analysis

## Results 📊

Empirical experiments conducted on six foundation models demonstrate superior performance compared to existing SOTA baselines.

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/agiresearch/A-mem.git
cd A-mem
```

2. Install dependencies:
Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

Install the package:
```bash
pip install .
```
For development, you can install it in editable mode:
```bash
pip install -e .
```

3. Usage Examples 💡

Here's how to use the Agentic Memory system for basic operations:

```python
from agentic_memory.memory_system import AgenticMemorySystem

# Initialize the memory system 🚀
memory_system = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',  # Embedding model for ChromaDB
    llm_backend="openai",           # LLM backend (openai/ollama)
    llm_model="gpt-4o-mini"         # LLM model name
)

# Add Memories with Automatic LLM Analysis ✨
# Simple addition - LLM automatically generates keywords, context, and tags
memory_id1 = memory_system.add_note(
    "Machine learning algorithms use neural networks to process complex datasets and identify patterns."
)

# Check the automatically generated metadata
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

# Enhanced Retrieval with Metadata 🔍
# The system now uses generated metadata for better semantic search
results = memory_system.search("artificial intelligence data processing", k=3)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Keywords: {result['keywords']}")
    print(f"Tags: {result['tags']}")
    print(f"Relevance Score: {result.get('score', 'N/A')}")
    print("---")

# Alternative search methods
results = memory_system.search_agentic("neural networks", k=5)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Tags: {result['tags']}")
    print("---")

# Update Memories 🔄
memory_system.update(memory_id1, content="Updated: Deep learning neural networks for pattern recognition")

# Delete Memories ❌
memory_system.delete(memory_id3)

# Memory Evolution 🧬
# The system automatically evolves memories by:
# 1. Using LLM to analyze content and generate semantic metadata
# 2. Finding relationships using enhanced ChromaDB embeddings (content + metadata)
# 3. Updating tags, context, and connections based on related memories
# 4. Creating semantic links between memories
# This happens automatically when adding or updating memories!
```

### Advanced Features 🌟

1. **Intelligent LLM Analysis** 🧠
   - Automatic keyword extraction from content
   - Context generation based on semantic understanding
   - Smart tag assignment for categorization
   - Seamless integration with OpenAI and Ollama backends

2. **Enhanced ChromaDB Vector Storage** 📦
   - Embedding generation using content + metadata for superior semantic search
   - Fast similarity search leveraging both content and generated attributes
   - Automatic metadata serialization and handling
   - Persistent memory storage with rich semantic information

3. **Memory Evolution** 🧬
   - Automatically analyzes content relationships using LLM-generated metadata
   - Updates tags and context based on related memories
   - Creates semantic connections between memories
   - Dynamic memory organization with improved accuracy

4. **Flexible Metadata Management** 📋
   - Auto-generation when not provided (keywords, context, tags)
   - Manual override support for custom metadata
   - Partial metadata completion (LLM fills missing attributes)
   - Timestamp tracking and retrieval count monitoring

5. **Multiple LLM Backends** 🤖
   - OpenAI (GPT-4, GPT-4o-mini, GPT-3.5)
   - Ollama (for local deployment)
   - Configurable model selection for analysis and evolution

### Best Practices 💪

1. **Memory Creation** ✨:
   - Provide clear, descriptive content for better LLM analysis
   - Let the system auto-generate metadata for optimal semantic richness
   - Use partial metadata provision when you want to control specific attributes
   - Provide manual metadata only when you need precise control

2. **Memory Retrieval** 🔍:
   - Leverage semantic search with natural language queries
   - Use specific domain terminology that matches generated keywords
   - Adjust 'k' parameter based on needed results (typically 3-10)
   - Take advantage of enhanced retrieval using both content and metadata

3. **Memory Evolution** 🧬:
   - Allow automatic evolution to maximize memory organization
   - Review LLM-generated metadata periodically for accuracy
   - Use consistent domain-specific terminology for better clustering
   - Monitor memory connections to understand knowledge relationships

4. **LLM Integration** 🤖:
   - Ensure API keys are properly configured for your chosen backend
   - Use gpt-4o-mini for cost-effective analysis or gpt-4 for higher quality
   - Consider Ollama for local deployment and privacy requirements
   - Monitor LLM usage for cost management

5. **Error Handling** ⚠️:
   - Always check return values and handle None responses
   - Handle potential KeyError for non-existent memories
   - Use try-except blocks for LLM operations (network/API failures)
   - Implement fallback behavior when LLM analysis fails

## Citation 📚

If you use this code in your research, please cite our work:

```bibtex
@article{xu2025mem,
  title={A-mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2502.12110},
  year={2025}
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.
