from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import pickle
from nltk.tokenize import word_tokenize
import os
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def simple_tokenize(text):
    return word_tokenize(text)


class AzureOpenAIEmbeddingFunction:
    """ChromaDB-compatible embedding function backed by Azure OpenAI.

    Args:
        model: Azure OpenAI embedding deployment name (e.g. "text-embedding-3-small").
        api_key: Azure OpenAI API key. Falls back to AZURE_OPENAI_API_KEY env var.
        azure_endpoint: Azure OpenAI endpoint URL. Falls back to AZURE_OPENAI_ENDPOINT env var.
        api_version: Azure OpenAI API version. Falls back to AZURE_OPENAI_API_VERSION env var.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("openai package not found. Install it with: pip install openai")

        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        if not api_key:
            raise ValueError("Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY environment variable.")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT environment variable.")

        self.model = model
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    def name(self) -> str:
        return "azure_openai_embedding"

    def _embed(self, input: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=input)
        return [item.embedding for item in response.data]

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self._embed(input if isinstance(input, list) else [input])


class ChromaRetriever:
    """Vector database retrieval using ChromaDB.

    Supports two embedding providers:
    - ``"sentence_transformer"`` (default): local SentenceTransformer model.
    - ``"azure_openai"``: Azure OpenAI embedding model (e.g. ``text-embedding-3-small``).
    """

    def __init__(
        self,
        collection_name: str = "memories",
        model_name: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence_transformer",
        azure_embedding_model: str = "text-embedding-3-small",
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
    ):
        """Initialize ChromaDB retriever.

        Args:
            collection_name: Name of the ChromaDB collection.
            model_name: SentenceTransformer model name (used when embedding_provider="sentence_transformer").
            embedding_provider: ``"sentence_transformer"`` or ``"azure_openai"``.
            azure_embedding_model: Azure OpenAI embedding deployment name.
            azure_api_key: Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var).
            azure_endpoint: Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT env var).
            azure_api_version: Azure OpenAI API version (or set AZURE_OPENAI_API_VERSION env var).
        """
        self.client = chromadb.Client(Settings(allow_reset=True))

        if embedding_provider == "azure_openai":
            self.embedding_function = AzureOpenAIEmbeddingFunction(
                model=azure_embedding_model,
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
        else:
            self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)

        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )
        
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB with enhanced embedding using metadata.
        
        Args:
            document: Text content to add
            metadata: Dictionary of metadata including keywords, tags, context
            doc_id: Unique identifier for the document
        """
        # Build enhanced document content including semantic metadata
        enhanced_document = document
        
        # Add context information
        if 'context' in metadata and metadata['context'] != "General":
            enhanced_document += f" context: {metadata['context']}"
        
        # Add keywords information    
        if 'keywords' in metadata and metadata['keywords']:
            keywords = metadata['keywords'] if isinstance(metadata['keywords'], list) else json.loads(metadata['keywords'])
            if keywords:
                enhanced_document += f" keywords: {', '.join(keywords)}"
        
        # Add tags information
        if 'tags' in metadata and metadata['tags']:
            tags = metadata['tags'] if isinstance(metadata['tags'], list) else json.loads(metadata['tags'])
            if tags:
                enhanced_document += f" tags: {', '.join(tags)}"
        
        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
        
        # Store enhanced document content for better embedding
        processed_metadata['enhanced_content'] = enhanced_document
                
        # Use enhanced document content for embedding generation
        self.collection.add(
            documents=[enhanced_document],
            metadatas=[processed_metadata],
            ids=[doc_id]
        )
        
    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.
        
        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])
        
    def search(self, query: str, k: int = 5):
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Convert string metadata back to original types
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            # First level is a list with one item per query
            for i in range(len(results['metadatas'])):
                # Second level is a list of metadata dicts for each result
                if isinstance(results['metadatas'][i], list):
                    for j in range(len(results['metadatas'][i])):
                        # Process each metadata dict
                        if isinstance(results['metadatas'][i][j], dict):
                            metadata = results['metadatas'][i][j]
                            for key, value in metadata.items():
                                try:
                                    # Try to parse JSON for lists and dicts
                                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                        metadata[key] = json.loads(value)
                                    # Convert numeric strings back to numbers
                                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                        if '.' in value:
                                            metadata[key] = float(value)
                                        else:
                                            metadata[key] = int(value)
                                except (json.JSONDecodeError, ValueError):
                                    # If parsing fails, keep the original string
                                    pass
                        
        return results
