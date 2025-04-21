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
        self.pc = Pinecone(api_key=api_key)
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
