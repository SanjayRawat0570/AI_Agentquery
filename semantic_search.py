"""
PHASE 2.3: Vector DB Enhancement
Advanced semantic search with ChromaDB
"""

import logging
import os
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Enhanced semantic search using ChromaDB
    Provides document ranking, similarity scoring, and metadata filtering
    """
    
    def __init__(self, db_path: str = "chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self.available = False
        self.client = None
        self.collections = {}
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Initialize ChromaDB with persistence
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=db_path,
                anonymized_telemetry=False
            )
            
            self.client = chromadb.Client(settings)
            self.available = True
            logger.info(f"SemanticSearchEngine initialized with ChromaDB at {db_path}")
        
        except ImportError:
            logger.warning("chromadb not installed. Install with: pip install chromadb")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
    
    def create_collection(self, name: str, metadata: Optional[Dict] = None) -> bool:
        """Create a new collection for semantic search"""
        if not self.available:
            logger.warning("ChromaDB not available")
            return False
        
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata=metadata or {},
                embedding_function=None  # Use default embedding
            )
            self.collections[name] = collection
            logger.info(f"Created/retrieved collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            return False
    
    def add_documents(self, collection_name: str, documents: List[str],
                     metadatas: Optional[List[Dict]] = None,
                     ids: Optional[List[str]] = None) -> bool:
        """
        Add documents to a collection for semantic indexing
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional custom IDs for documents
            
        Returns:
            True if successful
        """
        if not self.available or collection_name not in self.collections:
            logger.warning(f"Collection {collection_name} not available")
            return False
        
        try:
            collection = self.collections[collection_name]
            
            if ids is None:
                ids = [f"doc-{i}" for i in range(len(documents))]
            
            if metadatas is None:
                metadatas = [{} for _ in documents]
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to {collection_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def search(self, collection_name: str, query: str, top_k: int = 5,
              where: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Semantic search in a collection
        
        Args:
            collection_name: Name of the collection to search
            query: Search query string
            top_k: Number of top results to return
            where: Optional filter conditions
            
        Returns:
            Dict with results, scores, and metadata
        """
        if not self.available or collection_name not in self.collections:
            logger.warning(f"Collection {collection_name} not available")
            return {"success": False, "error": "Collection not available"}
        
        try:
            collection = self.collections[collection_name]
            
            # Query with optional filtering
            if where:
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=where,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
            
            # Process and rank results
            ranked_results = self._rank_results(results, query)
            
            logger.info(f"Search in {collection_name} returned {len(ranked_results)} results")
            
            return {
                "success": True,
                "query": query,
                "collection": collection_name,
                "results": ranked_results,
                "total_results": len(ranked_results)
            }
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"success": False, "error": str(e)}
    
    def hybrid_search(self, collection_name: str, query: str, keywords: Optional[List[str]] = None,
                     top_k: int = 5) -> Dict[str, Any]:
        """
        Hybrid search combining semantic and keyword matching
        
        Args:
            collection_name: Collection to search
            query: Semantic query string
            keywords: Optional keywords for additional filtering
            top_k: Number of results
            
        Returns:
            Combined results from semantic + keyword search
        """
        # First do semantic search
        semantic_results = self.search(collection_name, query, top_k * 2)
        
        if not semantic_results["success"]:
            return semantic_results
        
        results = semantic_results["results"]
        
        # Optional keyword filtering
        if keywords:
            filtered_results = []
            query_lower = query.lower()
            keyword_lower = [k.lower() for k in keywords]
            
            for result in results:
                doc_lower = result["document"].lower()
                # Boost score if keywords found
                keyword_matches = sum(1 for k in keyword_lower if k in doc_lower)
                if keyword_matches > 0:
                    result["keyword_boost"] = keyword_matches * 0.1
                    result["similarity_score"] += result["keyword_boost"]
                
                filtered_results.append(result)
            
            results = sorted(filtered_results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
        
        return {
            "success": True,
            "query": query,
            "keywords": keywords,
            "collection": collection_name,
            "results": results,
            "total_results": len(results),
            "search_type": "hybrid"
        }
    
    def _rank_results(self, raw_results: Dict, query: str) -> List[Dict]:
        """Rank and format raw ChromaDB results"""
        ranked = []
        
        if not raw_results["documents"] or not raw_results["documents"][0]:
            return ranked
        
        docs = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0] if raw_results["metadatas"] else [{}] * len(docs)
        distances = raw_results["distances"][0] if raw_results["distances"] else [1.0] * len(docs)
        
        for i, (doc, metadata, distance) in enumerate(zip(docs, metadatas, distances)):
            # Convert distance to similarity (inverse of distance for similarity metrics)
            similarity_score = 1.0 - (distance / 2.0) if distance <= 2.0 else 0.0
            
            ranked.append({
                "rank": i + 1,
                "document": doc,
                "metadata": metadata or {},
                "similarity_score": round(similarity_score, 3),
                "distance": round(distance, 3),
                "relevance": "high" if similarity_score > 0.7 else "medium" if similarity_score > 0.5 else "low"
            })
        
        return ranked
    
    def update_document(self, collection_name: str, doc_id: str, document: str,
                       metadata: Optional[Dict] = None) -> bool:
        """Update an existing document"""
        if not self.available or collection_name not in self.collections:
            return False
        
        try:
            collection = self.collections[collection_name]
            collection.update(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata or {}]
            )
            logger.info(f"Updated document {doc_id} in {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            return False
    
    def delete_document(self, collection_name: str, doc_id: str) -> bool:
        """Delete a document from collection"""
        if not self.available or collection_name not in self.collections:
            return False
        
        try:
            collection = self.collections[collection_name]
            collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id} from {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection"""
        if not self.available or collection_name not in self.collections:
            return {"success": False, "error": "Collection not available"}
        
        try:
            collection = self.collections[collection_name]
            count = collection.count()
            
            return {
                "success": True,
                "collection_name": collection_name,
                "document_count": count,
                "model_name": self.model_name,
                "metadata": collection.metadata or {}
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"success": False, "error": str(e)}
    
    def persist(self) -> bool:
        """Persist collections to disk"""
        if not self.available:
            return False
        
        try:
            self.client.persist()
            logger.info("Collections persisted to disk")
            return True
        except Exception as e:
            logger.error(f"Failed to persist collections: {e}")
            return False


# Global semantic search engine instance
semantic_search_engine = SemanticSearchEngine()
