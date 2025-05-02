import logging
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
import time
import concurrent.futures
import re

logger = logging.getLogger(__name__)

class PineconeClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PineconeClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("Pinecone API key is not set")

        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'asset-support-index')
        self.pc = Pinecone(api_key=api_key, environment=os.getenv("PINECONE_ENVIRONMENT"))
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.index = self.pc.Index(self.index_name)
        model_name = os.getenv('EMBEDDING_MODEL', 'all-mpnet-base-v2')
        self.embedding_model = SentenceTransformer(model_name)
        self.query_cache = {}
        self.METADATA_SIZE_LIMIT = 40000
        logger.info(f"Initialized PineconeClient with model {model_name}")

    @lru_cache(maxsize=1024)
    def generate_embedding(self, text):
        if not text or not text.strip():
            return None
        emb = self.embedding_model.encode(text)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        return (emb / norm).tolist()

    def store_document_chunks(self, chunks, asset_id, document_id, chunk_size=500, overlap=50):
        """
        Store chunks in Pinecone:
        - Supports raw text (auto-chunking) or pre-split list
        - Overlapping sliding window chunking for semantic coherence
        - Parallel embedding generation
        - Metadata truncation to fit limits
        """
        if isinstance(chunks, str):
            text = chunks
            words = text.split()
            chunks = []
            step = chunk_size - overlap
            for i in range(0, len(words), step):
                window = words[i:i + chunk_size]
                if not window:
                    break
                chunks.append(" ".join(window))

        if not chunks:
            logger.warning("No chunks to store")
            return False

        asset_str = str(asset_id)
        vectors = []
        truncated = 0

        # Generate embeddings in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self.generate_embedding, chunk): idx for idx, chunk in enumerate(chunks)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                text = chunks[idx]
                emb = None
                try:
                    emb = future.result(timeout=5)
                except Exception as e:
                    logger.warning(f"Embedding failed for chunk {idx}: {e}")
                if emb is None:
                    continue

                meta = {"asset_id": asset_str, "document_id": str(document_id), "chunk_index": idx}
                meta_bytes = len(str(meta).encode('utf-8'))
                available = self.METADATA_SIZE_LIMIT - meta_bytes - 100
                text_bytes = text.encode('utf-8')
                if len(text_bytes) > available:
                    truncated_text = text_bytes[:available].decode('utf-8', errors='ignore')
                    meta.update({"text": truncated_text, "is_truncated": True})
                    truncated += 1
                else:
                    meta["text"] = text
                vector_id = f"{document_id}_{idx}"
                vectors.append((vector_id, emb, meta))

        if not vectors:
            logger.warning("No valid vectors after embedding generation")
            return False
        if truncated:
            logger.info(f"Truncated {truncated}/{len(vectors)} chunks due to metadata size limits")

        # Upsert in manageable batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            ids, embs, metas = zip(*batch)
            try:
                self.index.upsert(vectors=list(zip(ids, embs, metas)))
            except Exception as e:
                logger.error(f"Upsert batch failed: {e}")
        logger.info(f"Successfully upserted {len(vectors)} chunks for document {document_id}")
        return True

    def debug_index_contents(self, asset_id):
        resp = self.index.query(
            vector=[0]*768,
            filter={"asset_id": str(asset_id)},
            top_k=100,
            include_metadata=True
        )
        return [{"metadata": m.metadata, "score": m.score} for m in resp.matches]

    def _get_emergency_fallback(self, asset_id, limit=3):
        key = f"fallback_{asset_id}_{limit}"
        if key in self.query_cache:
            return self.query_cache[key]
        resp = self.index.query(
            vector=[0]*768,
            filter={"asset_id": str(asset_id)},
            top_k=limit,
            include_metadata=True
        )
        fallback = [{"text": m.metadata.get("text",""), "score":0.1, "chunk_index": m.metadata.get("chunk_index",-1)} \
                    for m in resp.matches]
        self.query_cache[key] = fallback
        return fallback

    def get_fallback_chunks(self, asset_id, query=None, limit=5, methods=None):
        """
        Advanced fallback retrieval that uses multiple techniques to find relevant chunks
        when standard semantic search fails.
        
        Args:
            asset_id (str): The asset ID to search within
            query (str): The query text to find relevant chunks for
            limit (int): Number of chunks to return
            methods (list): List of fallback methods to try, defaults to all
                Options: 'keyword', 'ngram', 'expanded', 'contextual', 'random'
                
        Returns:
            list: Ranked list of relevant chunks with scores
        """
        if not methods:
            methods = ['keyword', 'ngram', 'expanded', 'contextual', 'random']
        
        logger.info(f"Executing fallback retrieval with methods: {methods}")
        
        query_text = query if query else ""
        
        # Cache key for this query
        cache_key = f"deep_fallback_{asset_id}_{hash(query_text)}_{limit}_{'-'.join(methods)}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        all_results = []
        
        # Method 1: Keyword matching with tf-idf style scoring
        if 'keyword' in methods:
            keywords = set(re.findall(r"\w{3,}", query_text.lower()))
            if keywords:
                # Get a larger sample of chunks to analyze
                sample_chunks = self._fetch_asset_chunks(asset_id, limit=100)
                
                # Calculate inverse document frequency for weighting
                keyword_doc_count = Counter()
                for chunk in sample_chunks:
                    chunk_text = chunk.get('text', '').lower()
                    found_keywords = set()
                    for keyword in keywords:
                        if keyword in chunk_text:
                            found_keywords.add(keyword)
                    for keyword in found_keywords:
                        keyword_doc_count[keyword] += 1
                
                # Calculate tf-idf style scores
                keyword_results = []
                for chunk in sample_chunks:
                    text = chunk.get('text', '')
                    chunk_text_lower = text.lower()
                    
                    score = 0
                    for keyword in keywords:
                        # Skip very common words that appear in most documents
                        if keyword_doc_count[keyword] > len(sample_chunks) * 0.8:
                            continue
                            
                        # Term frequency
                        tf = chunk_text_lower.count(keyword)
                        if tf > 0:
                            # Inverse document frequency component
                            idf = np.log(len(sample_chunks) / (1 + keyword_doc_count[keyword]))
                            score += tf * idf
                    
                    # Normalize by text length to avoid favoring longer chunks
                    if text:
                        score = score / (np.log(1 + len(text.split())))
                    
                    if score > 0:
                        keyword_results.append({
                            "text": text,
                            "score": 0.3 + score * 0.2,  # Scale to be comparable with semantic scores
                            "chunk_index": chunk.get("chunk_index", -1),
                            "method": "keyword"
                        })
                
                keyword_results.sort(key=lambda x: x['score'], reverse=True)
                all_results.extend(keyword_results[:limit])
        
        # Method 2: N-gram matching for phrase queries
        if 'ngram' in methods and len(query_text.split()) > 2:
            # Extract important phrases from the query
            phrases = []
            words = query_text.split()
            for i in range(len(words) - 1):
                phrases.append(f"{words[i]} {words[i+1]}")
            
            if len(words) >= 3:
                for i in range(len(words) - 2):
                    phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
            
            # Get chunks and score by phrase matches
            sample_chunks = self._fetch_asset_chunks(asset_id, limit=50)
            ngram_results = []
            
            for chunk in sample_chunks:
                text = chunk.get('text', '')
                if not text:
                    continue
                    
                matches = 0
                for phrase in phrases:
                    if phrase.lower() in text.lower():
                        matches += 1
                
                if matches > 0:
                    score = 0.25 + (matches / len(phrases)) * 0.25
                    ngram_results.append({
                        "text": text,
                        "score": score,
                        "chunk_index": chunk.get("chunk_index", -1),
                        "method": "ngram"
                    })
            
            ngram_results.sort(key=lambda x: x['score'], reverse=True)
            all_results.extend(ngram_results[:limit])
        
        # Method 3: Expanded query using related terms
        if 'expanded' in methods:
            # Expand the query with related terms/synonyms
            expanded_terms = self._expand_query_terms(query_text)
            
            if expanded_terms:
                try:
                    expanded_query = query_text + " " + " ".join(expanded_terms)
                    emb = self.generate_embedding(expanded_query)
                    
                    if emb:
                        resp = self.index.query(
                            vector=emb,
                            filter={"asset_id": str(asset_id)},
                            top_k=limit,
                            include_metadata=True
                        )
                        
                        expanded_results = [
                            {
                                "text": m.metadata.get("text", ""), 
                                "score": m.score * 0.9,  # Slightly lower confidence
                                "chunk_index": m.metadata.get("chunk_index", -1),
                                "method": "expanded"
                            }
                            for m in resp.matches if m.score > 0.3
                        ]
                        all_results.extend(expanded_results)
                except Exception as e:
                    logger.warning(f"Expanded query search failed: {e}")
        
        # Method 4: Contextual search by retrieving adjacent chunks
        if 'contextual' in methods:
            # First get some base chunks
            base_chunks = []
            for result in all_results[:5]:  # Use top results from other methods
                if 'chunk_index' in result and result['chunk_index'] >= 0:
                    base_chunks.append(result['chunk_index'])
            
            if base_chunks:
                # Get adjacent chunks
                adjacent_indices = set()
                for idx in base_chunks:
                    adjacent_indices.add(idx - 1)
                    adjacent_indices.add(idx + 1)
                
                adjacent_indices = adjacent_indices - set(base_chunks)
                if adjacent_indices:
                    try:
                        # Fetch these specific chunks
                        contextual_results = []
                        sample_chunks = self._fetch_asset_chunks(asset_id, limit=100)
                        
                        for chunk in sample_chunks:
                            if chunk.get('chunk_index') in adjacent_indices:
                                contextual_results.append({
                                    "text": chunk.get('text', ''),
                                    "score": 0.4,  # Contextual chunks get medium confidence
                                    "chunk_index": chunk.get('chunk_index', -1),
                                    "method": "contextual"
                                })
                        
                        all_results.extend(contextual_results[:limit//2])
                    except Exception as e:
                        logger.warning(f"Contextual search failed: {e}")
        
        # Method 5: Random sampling as last resort
        if 'random' in methods and len(all_results) < limit:
            try:
                needed = limit - len(all_results)
                random_chunks = self._get_emergency_fallback(asset_id, limit=needed * 2)
                
                # Add method identifier
                for chunk in random_chunks:
                    chunk['method'] = 'random'
                    
                all_results.extend(random_chunks[:needed])
            except Exception as e:
                logger.warning(f"Random sampling failed: {e}")
        
        # Deduplicate by chunk_index
        seen_indices = set()
        unique_results = []
        
        for result in sorted(all_results, key=lambda x: x.get('score', 0), reverse=True):
            chunk_index = result.get('chunk_index')
            if chunk_index not in seen_indices:
                seen_indices.add(chunk_index)
                unique_results.append(result)
                if len(unique_results) >= limit:
                    break
        
        # Cache and return
        self.query_cache[cache_key] = unique_results[:limit]
        logger.info(f"Fallback retrieval found {len(unique_results)} chunks using methods: {methods}")
        return unique_results[:limit]
    
    def _fetch_asset_chunks(self, asset_id, limit=50):
        """Helper to fetch chunks for an asset"""
        try:
            resp = self.index.query(
                vector=[0]*768,  # Zero vector to get random chunks
                filter={"asset_id": str(asset_id)},
                top_k=limit,
                include_metadata=True
            )
            return [
                {"text": m.metadata.get("text", ""), 
                 "chunk_index": m.metadata.get("chunk_index", -1)} 
                for m in resp.matches
            ]
        except Exception as e:
            logger.error(f"Error fetching asset chunks: {e}")
            return []
    
    def _expand_query_terms(self, query_text):
        """
        Generate expanded terms for the query to broaden search
        Returns list of additional search terms
        """
        # Simple expansion for common terms - would be better with a thesaurus
        expansions = {
            "error": ["exception", "failure", "issue", "problem", "bug"],
            "install": ["setup", "configure", "deploy"],
            "api": ["endpoint", "service", "interface"],
            "data": ["information", "records", "content"],
            "user": ["account", "profile", "customer"],
            "config": ["settings", "options", "preferences"],
            "database": ["db", "storage", "repository"],
            "auth": ["authentication", "login", "credentials"],
            "file": ["document", "attachment"],
            "update": ["upgrade", "patch", "change"],
            "delete": ["remove", "erase"],
            "add": ["create", "insert"],
            "function": ["method", "procedure", "routine"],
            "variable": ["parameter", "argument", "attribute"],
            "library": ["package", "module", "dependency"],
            "event": ["trigger", "callback", "action"],
            "version": ["release", "build"],
            "test": ["check", "verify", "validate"],
            "cloud": ["hosted", "remote", "service"]
        }
        
        expanded = []
        words = set(re.findall(r"\w+", query_text.lower()))
        
        for word in words:
            if word in expansions:
                expanded.extend(expansions[word][:2])  # Limit to 2 expansions per word
                
        return expanded[:5]  # Limit total expansions

    def query_similar_chunks(self, query_text, asset_id, top_k=5, similarity_threshold=0.4):
        """
        Retrieve top_k chunks for query_text using semantic + keyword fallback.
        """
        if not query_text:
            return []
        key = f"sim_{asset_id}_{hash(query_text)}_{top_k}_{similarity_threshold}"
        if key in self.query_cache:
            return self.query_cache[key]

        emb = self.generate_embedding(query_text)
        if emb is None:
            return self._get_emergency_fallback(asset_id, top_k)

        try:
            resp = self.index.query(
                vector=emb,
                filter={"asset_id": str(asset_id)},
                top_k=top_k*3,
                include_metadata=True
            )
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return self._get_emergency_fallback(asset_id, top_k)

        sem = sorted([
            {"text": m.metadata.get("text",""), "score": m.score, "chunk_index": m.metadata.get("chunk_index",-1)}
            for m in resp.matches
        ], key=lambda x: x['score'], reverse=True)
        filtered = [c for c in sem if c['score'] >= similarity_threshold]

        if len(filtered) < top_k:
            logger.info("Keyword fallback engaged")
            terms = set(re.findall(r"\w{4,}", query_text.lower()))
            debug = self.debug_index_contents(asset_id)
            kw = []
            for item in debug:
                txt = item['metadata'].get('text','')
                words = set(txt.lower().split())
                common = terms & words
                if common:
                    score = 0.2 + (len(common)/len(terms))*0.5
                    kw.append({"text": txt, "score": score, "chunk_index": item['metadata'].get('chunk_index',-1)})
            combined = (filtered + kw)
            seen = set(); out = []
            for c in sorted(combined, key=lambda x: x['score'], reverse=True):
                if c['chunk_index'] not in seen:
                    seen.add(c['chunk_index']); out.append(c)
                if len(out) >= top_k:
                    break
            filtered = out

        result = filtered[:top_k] if filtered else self._get_emergency_fallback(asset_id, top_k)
        self.query_cache[key] = result
        return result
    def delete_document(self, document_id: str, asset_id: str):
        """
        Delete all vectors associated with a given document by:
        1. Querying vectors for the given asset.
        2. Filtering matches where metadata.document_id == document_id.
        3. Deleting those vectors by their IDs.
        """
        try:
            # Use the correct vector dimension (768) for the dummy query vector.
            dummy_vector = [0] * 768
            results = self.index.query(
                vector=dummy_vector,
                filter={"asset_id": str(asset_id)},
                top_k=1000,
                include_metadata=True
            )

            # Extract IDs for matches with the specified document_id.
            ids_to_delete = [
                match.id
                for match in results.matches
                if match.metadata.get("document_id") == document_id
            ]

            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} vectors for document {document_id}")
            else:
                logger.warning(f"No vectors found for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
