import logging
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
import time
import concurrent.futures

logger = logging.getLogger(__name__)

class PineconeClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PineconeClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize Pinecone client and load an optimized embedding model."""
        try:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("Pinecone API key is not set")
            
            self.index_name = os.getenv('PINECONE_INDEX_NAME', 'asset-support-index')
            self.pc = Pinecone(api_key=api_key)
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # Using a lighter model dimension (MiniLM)
                    metric="cosine",
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            self.index = self.pc.Index(self.index_name)
            
            # Use a lighter and faster embedding model by default.
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
            
            # In-memory cache for query results (key: query parameters, value: result)
            self.query_cache = {}
            
            # Define Pinecone metadata size limit (slightly below the actual limit for safety)
            self.METADATA_SIZE_LIMIT = 40000  # Actual limit is 40960 bytes
            
            logger.info(f"Pinecone client initialized successfully using model {model_name}")
        except Exception as e:
            logger.error(f"Pinecone initialization error: {str(e)}")
            raise

    @lru_cache(maxsize=1024)
    def generate_embedding(self, text):
        """Generate and cache embedding for a text segment."""
        try:
            if not text or not text.strip():
                logger.warning("Attempted to generate embedding for empty text")
                return None
            
            start_time = time.perf_counter()
            embedding = self.embedding_model.encode(text)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                logger.warning("Zero norm encountered during embedding generation")
                return None
            embedding = embedding / norm
            elapsed = time.perf_counter() - start_time
            logger.debug(f"Generated embedding in {elapsed:.4f} seconds for text: {text[:30]}...")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            return None

    def store_document_chunks(self, chunks, asset_id, document_id):
        """
        Store document chunks in Pinecone with robust error handling and
        metadata size limit management.
        """
        try:
            if not chunks:
                logger.warning("No chunks provided for embedding")
                return False
            
            vectors = []
            large_chunks_count = 0
            
            for i, chunk in enumerate(chunks):
                if not chunk or not chunk.strip():
                    logger.debug(f"Skipping empty chunk {i}")
                    continue
                
                vector_id = f"{document_id}_{i}"
                embedding = self.generate_embedding(chunk)
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for chunk {i}")
                    continue
                
                asset_id_str = str(asset_id)
                
                # Base metadata without text field
                base_metadata = {
                    "asset_id": asset_id_str,
                    "document_id": str(document_id),
                    "chunk_index": i,
                }
                
                # Estimate base metadata size
                base_metadata_size = len(str(base_metadata).encode('utf-8'))
                
                # Calculate available space for text
                text_size_limit = self.METADATA_SIZE_LIMIT - base_metadata_size - 100  # Extra buffer for safety
                
                # Check if text needs truncation
                chunk_bytes = chunk.encode('utf-8')
                chunk_size = len(chunk_bytes)
                
                metadata = base_metadata.copy()
                if chunk_size <= text_size_limit:
                    # Text fits within limit
                    metadata["text"] = chunk
                else:
                    # Text needs truncation
                    large_chunks_count += 1
                    # Truncate to byte limit then decode back to string
                    truncated_text = chunk_bytes[:text_size_limit].decode('utf-8', errors='ignore')
                    metadata["text"] = truncated_text
                    metadata["is_truncated"] = True
                    metadata["original_length"] = len(chunk)
                    metadata["truncated_length"] = len(truncated_text)
                    logger.info(f"Chunk {i} truncated from {len(chunk)} to {len(truncated_text)} characters due to metadata size limit")
                
                vectors.append((vector_id, embedding, metadata))
            
            if large_chunks_count > 0:
                logger.warning(f"Truncated {large_chunks_count} out of {len(vectors)} chunks due to metadata size limits")
            
            if vectors:
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i+batch_size]
                    ids, embeddings, metadatas = zip(*batch)
                    self.index.upsert(vectors=zip(ids, embeddings, metadatas))
                logger.info(f"Successfully stored {len(vectors)} chunks for document {document_id}")
                return True
            else:
                logger.warning("No valid vectors to store")
                return False
        
        except Exception as e:
            logger.error(f"Error storing document chunks: {str(e)}")
            return False

    def _generate_cache_key(self, query_text, asset_id, top_k, similarity_threshold):
        """Generate a cache key for query results."""
        return f"{query_text}:{asset_id}:{top_k}:{similarity_threshold}"
    
    def query_similar_chunks(self, query_text, asset_id, top_k=5, similarity_threshold=0.7):
        """
        Highly optimized query method to prevent timeouts and 504 errors.
        """
        try:
            asset_id_str = str(asset_id)
            cache_key = f"chunk_query_{asset_id_str}_{hash(query_text)}_{top_k}_{similarity_threshold}"
            
            # Check cache first with higher priority
            cached_result = self.query_cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached query results")
                return cached_result
            
            # Generate embedding with timeout
            start_time = time.perf_counter()
            query_embedding = None
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.generate_embedding, query_text)
                    query_embedding = future.result(timeout=3.0)  # Hard timeout for embedding generation
            except concurrent.futures.TimeoutError:
                logger.error("Embedding generation timed out")
                return self._get_emergency_fallback(asset_id, top_k)
                
            if query_embedding is None:
                logger.warning("Query embedding generation failed")
                return self._get_emergency_fallback(asset_id, top_k)
            
            # Super aggressive approach: very small initial fetch with short timeout
            initial_top_k = min(top_k, 2)
            chunks = []
            
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self.index.query,
                        vector=query_embedding,
                        filter={"asset_id": asset_id_str},
                        top_k=initial_top_k,
                        include_metadata=True
                    )
                    initial_results = future.result(timeout=5.0)  # Short timeout
                    
                    # Process initial results
                    for match in initial_results.matches:
                        score = match.score
                        if score >= similarity_threshold:
                            text = match.metadata.get("text", "")
                            chunk_info = {
                                "text": text,
                                "score": score,
                                "document_id": match.metadata.get("document_id", ""),
                                "chunk_index": match.metadata.get("chunk_index", -1),
                                "is_truncated": match.metadata.get("is_truncated", False)
                            }
                            chunks.append(chunk_info)
                    
                    # Only if initial fetch was fast (< 2 seconds), try for more
                    elapsed = time.perf_counter() - start_time
                    if elapsed < 2.0 and len(chunks) < top_k and top_k > initial_top_k:
                        remaining_top_k = min(top_k, 5)  # Cap at 5 to prevent timeouts
                        
                        additional_future = executor.submit(
                            self.index.query,
                            vector=query_embedding,
                            filter={"asset_id": asset_id_str},
                            top_k=remaining_top_k,
                            include_metadata=True
                        )
                        # Shorter timeout for additional results
                        additional_results = additional_future.result(timeout=4.0)
                        
                        existing_ids = {chunk["chunk_index"]: True for chunk in chunks}
                        for match in additional_results.matches[initial_top_k:]:
                            chunk_index = match.metadata.get("chunk_index", -1)
                            if chunk_index in existing_ids:
                                continue
                            
                            score = match.score
                            if score >= similarity_threshold:
                                text = match.metadata.get("text", "")
                                chunk_info = {
                                    "text": text,
                                    "score": score,
                                    "document_id": match.metadata.get("document_id", ""),
                                    "chunk_index": chunk_index,
                                    "is_truncated": match.metadata.get("is_truncated", False)
                                }
                                chunks.append(chunk_info)
            
            except concurrent.futures.TimeoutError:
                logger.warning("Pinecone query timed out, using what we have")
                if not chunks:
                    return self._get_emergency_fallback(asset_id, top_k)
            except Exception as e:
                logger.error(f"Error in optimized query: {str(e)}")
                if not chunks:
                    return self._get_emergency_fallback(asset_id, top_k)
            
            # Sort by relevance and cache even partial results
            chunks.sort(key=lambda x: x["score"], reverse=True)
            self.query_cache[cache_key] = chunks
            
            logger.info(f"Found {len(chunks)} similar chunks for asset {asset_id_str}")
            return chunks
        
        except Exception as e:
            logger.error(f"Critical error in query_similar_chunks: {str(e)}")
            return self._get_emergency_fallback(asset_id, top_k)

    def debug_index_contents(self, asset_id):
        """
        Debug method to inspect index contents for a specific asset.
        """
        try:
            asset_id_str = str(asset_id)
            results = self.index.query(
                vector=[0]*384,  # Dummy vector adjusted for 384 dimensions
                filter={"asset_id": asset_id_str},
                top_k=100,
                include_metadata=True
            )
            debug_info = []
            for match in results.matches:
                debug_info.append({
                    "id": match.id,
                    "metadata": match.metadata,
                    "score": match.score
                })
            logger.info(f"Debug: Found {len(debug_info)} vectors for asset {asset_id_str}")
            return debug_info
        
        except Exception as e:
            logger.error(f"Error in debug_index_contents: {str(e)}")
            return []
        
    def get_fallback_chunks(self, asset_id, query=None, limit=3):
        """
        Enhanced fallback retrieval with query-based filtering when possible.
        """
        try:
            asset_id_str = str(asset_id)
            logger.info(f"Fetching fallback chunks for asset: {asset_id_str}")
            
            # If we have a query, try to use it for basic keyword filtering
            filter_dict = {"asset_id": asset_id_str}
            
            # Create a zero vector with correct dimensionality
            zero_vector = [0] * 384
            
            # Use a lower top_k initially to speed up the query
            initial_limit = min(limit, 3)
            fallback_results = self.index.query(
                vector=zero_vector,
                filter=filter_dict,
                top_k=initial_limit,
                include_metadata=True
            )
            
            fallback_chunks = []
            for match in fallback_results.matches:
                chunk_info = {
                    "text": match.metadata.get("text", ""),
                    "score": 0.5,  # Default score for fallback results
                    "document_id": match.metadata.get("document_id", ""),
                    "chunk_index": match.metadata.get("chunk_index", -1),
                    "is_truncated": match.metadata.get("is_truncated", False)
                }
                fallback_chunks.append(chunk_info)
            
            # If we need more and limit is higher, fetch additional chunks
            if len(fallback_chunks) < limit and initial_limit < limit:
                try:
                    additional_results = self.index.query(
                        vector=zero_vector,
                        filter=filter_dict,
                        top_k=limit,
                        include_metadata=True
                    )
                    
                    for match in additional_results.matches[initial_limit:]:
                        chunk_info = {
                            "text": match.metadata.get("text", ""),
                            "score": 0.4,  # Lower score for additional fallback results
                            "document_id": match.metadata.get("document_id", ""),
                            "chunk_index": match.metadata.get("chunk_index", -1),
                            "is_truncated": match.metadata.get("is_truncated", False)
                        }
                        fallback_chunks.append(chunk_info)
                except Exception as e:
                    logger.warning(f"Error fetching additional fallback chunks: {str(e)}")

            logger.info(f"Fetched {len(fallback_chunks)} fallback chunks for asset {asset_id_str}")
            return fallback_chunks
        
        except Exception as e:
            logger.error(f"Error retrieving fallback chunks: {str(e)}")
            return []
        
    def delete_document(self, document_id: str, asset_id: str):
        """
        Delete all vectors associated with a given document by first fetching
        all vectors for the asset, filtering by document_id, and then deleting
        by the vector IDs.
        """
        try:
            # Query all vectors for the asset.
            results = self.index.query(
                vector=[0] * 384,  # Dummy vector for broad retrieval.
                filter={"asset_id": str(asset_id)},
                top_k=1000,  # Adjust if needed.
                include_metadata=True
            )
            # Extract the vector IDs where metadata.document_id matches.
            ids_to_delete = [
                match.id
                for match in results.matches
                if match.metadata.get("document_id") == document_id
            ]
            
            if ids_to_delete:
                # Delete vectors using their IDs (do not use metadata filtering).
                self.index.delete(ids=ids_to_delete)
                logger.info(f"Deleted vectors for document {document_id} from Pinecone")
            else:
                logger.warning(f"No vectors found for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    def _get_emergency_fallback(self, asset_id, limit=3):
        """Emergency fallback when everything else fails"""
        try:
            # Use a cached zero vector query if possible
            cache_key = f"emergency_fallback_{asset_id}_{limit}"
            cached_fallback = self.query_cache.get(cache_key)
            if cached_fallback:
                logger.info("Using cached emergency fallback")
                return cached_fallback
                
            zero_vector = [0] * 384
            fallback_results = []
            
            # Just try to get ANY results with a very short timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.index.query,
                    vector=zero_vector,
                    filter={"asset_id": str(asset_id)},
                    top_k=limit,
                    include_metadata=True
                )
                results = future.result(timeout=3.0)
                
                for match in results.matches:
                    chunk_info = {
                        "text": match.metadata.get("text", ""),
                        "score": 0.5,
                        "document_id": match.metadata.get("document_id", ""),
                        "chunk_index": match.metadata.get("chunk_index", -1),
                        "is_truncated": match.metadata.get("is_truncated", False)
                    }
                    fallback_results.append(chunk_info)
            
            # Cache emergency fallback for longer (10 minutes)
            self.query_cache[cache_key] = fallback_results
            return fallback_results
            
        except Exception:
            # Last resort: return empty list
            logger.error("Emergency fallback failed - returning empty list")
            return []