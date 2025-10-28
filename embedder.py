from sentence_transformers import SentenceTransformer
from sentence_transformers.sparse_encoder import SparseEncoder
import torch
import os
import re
from tqdm import tqdm
import numpy as np
import chromadb
import json
import time


"""
Currently only do dense encoding
Sparse encoding related functions are placeholders
"""

DENSE_EMBEDDER_MODEL = "BAAI/bge-base-zh-v1.5"
SPARSE_EMBEDDER_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"


class TextCleaner:
    def __init__(self, lowercase=False, remove_urls=True, normalize_space=True):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.normalize_space = normalize_space

    def clean(self, text: str) -> str:
        text = text.strip()
        if self.lowercase:
            text = text.lower()
        if self.remove_urls:
            text = re.sub(r"http\S+", "", text)
        if self.normalize_space:
            text = re.sub(r"\s+", " ", text)
        return text


def join_chunk_text(text_chunk):
    # support chunk (list of sentences) processing
    if isinstance(text_chunk, list):
        return "\n".join(text_chunk)
    return text_chunk


class DenseTextEncoder:
    """
    output: numpy array
    """
    def __init__(self, model_name, normalize=True, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.cleaner = TextCleaner()

    def _prepare_texts(self, texts):
        """Support single string, list[str], or list[list[str]]"""
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, list):
            if all(isinstance(t, str) for t in texts):
                texts = [join_chunk_text(t) if isinstance(t, list) else t for t in texts]
            elif all(isinstance(t, list) for t in texts):
                texts = [join_chunk_text(t) for t in texts]
            else:
                raise ValueError("Input list must contain only str or list[str].")
        else:
            raise ValueError("Input must be str or list.")
        cleaned = [self.cleaner.clean(t) for t in texts]
        return cleaned

    def encode_document(self, texts):
        cleaned = self._prepare_texts(texts)
        output = self.model.encode_document(cleaned, convert_to_numpy=True, normalize_embeddings=self.normalize)
        return output

    def encode_query(self, texts):
        cleaned = self._prepare_texts(texts)
        output = self.model.encode_query(cleaned, convert_to_numpy=True, normalize_embeddings=self.normalize)
        return output


class SparseTextEncoder:
    """
    output: torch tensor
    """
    def __init__(self, model_name, device=None):
        self.device = device or "cpu"
        self.encoder = SparseEncoder(model_name, device=self.device)
        self.cleaner = TextCleaner()

    def _prepare_texts(self, texts):
        """Support single string, list[str], or list[list[str]]"""
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, list):
            if all(isinstance(t, str) for t in texts):
                texts = [join_chunk_text(t) if isinstance(t, list) else t for t in texts]
            elif all(isinstance(t, list) for t in texts):
                texts = [join_chunk_text(t) for t in texts]
            else:
                raise ValueError("Input list must contain only str or list[str].")
        else:
            raise ValueError("Input must be str or list.")
        cleaned = [self.cleaner.clean(t) for t in texts]
        return cleaned

    def encode_document(self, texts):
        """Encode for corpus indexing"""
        cleaned = self._prepare_texts(texts)
        return self.encoder.encode_document(cleaned)

    def encode_query(self, texts):
        """Encode for query retrieval"""
        cleaned = self._prepare_texts(texts)
        return self.encoder.encode_query(cleaned)

def read_input(source):
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    else:
        return [source]


def encode_chunks_with_metadata(chunks, dense_encoder, sparse_encoder):
    """
    :param chunks: [{'text': [...], 'chunk_id': ..., ...metadata}, ...]
    :param dense_encoder: dense encoder model
    :param sparse_encoder: sparse encoder model
    :return:
    {
      "chunk_id": "9b28e9938292486e9a61f2d1787bb828",
      "dense_embedding": np.array([...]),
      "sparse_embedding": torch.sparse.Tensor(...),
      "text": "友希那: ...\n莉莎: ...",
      "eventName": "连结思绪的未竟之歌",
      "chapterTitle": "序章: 古旧的磁带",
      "story_type": "event",
      # ...other metadata
    }
    """
    text = [join_chunk_text(chunk["text"]) for chunk in chunks]
    dense_vecs = dense_encoder.encode_document(text)
    # placeholder, skip sparse encoding for now
    #sparse_vecs = sparse_encoder.encode_document(text)
    result = []
    for i, chunk in enumerate(chunks):

        # placeholder, skip sparse encoding for now
        #sparse_i = sparse_vecs[i]
        #if isinstance(sparse_i, torch.Tensor) and sparse_i.is_sparse:
        #    sparse_i = sparse_i.coalesce()

        result.append({
            "chunk_id": chunk.get("chunk_id"),
            "dense_embedding": dense_vecs[i],
            # placeholder, skip sparse encoding for now
            "sparse_embedding": None,
            #"sparse_embedding": sparse_vecs[i],
            "text": text[i],
            "eventName": chunk.get("eventName"),
            "chapterTitle": chunk.get("chapterTitle"),
            "story_type": chunk.get("story_type"),
            "start_idx": chunk.get("start_idx"),
            "end_idx": chunk.get("end_idx"),
        })
    return result

# save dense embedding to chroma vector database
def save_chunks_to_chroma(embedded_chunk, collection, batch_size=64):
    ids = []
    documents = []
    embeddings = []
    metadata = []

    for entry in embedded_chunk:
        ids.append(entry["chunk_id"])
        documents.append(entry["text"])
        embeddings.append(
            entry["dense_embedding"].tolist() if isinstance(entry["dense_embedding"], np.ndarray) else entry[
                "dense_embedding"])
        # currently do not store sparse embedding to chroma
        meta = {k: v for k, v in entry.items() if k not in ["chunk_id", "dense_embedding","sparse_embedding", "text"]}
        metadata.append(meta)

    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadata[i:i + batch_size]
        )
    print(f"saved {len(ids)} chunks to {collection.name}")

def read_jsonl_in_batches(file_path, batch_size=64):
    batch = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            if line.strip():
                batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


if __name__ == "__main__":
    chunk_files = [
        "./chunks/band_chunks.jsonl",
        "./chunks/card_chunks.jsonl",
        "./chunks/event_chunks.jsonl",
        "./chunks/main_chunks.jsonl"
    ]

    dense_encoder = DenseTextEncoder(DENSE_EMBEDDER_MODEL)
    sparse_encoder = SparseTextEncoder(SPARSE_EMBEDDER_MODEL)

    # init databases
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("bangdream_dense")

    start_time = time.time()
    for file_path in chunk_files:

        with open(file_path, 'r', encoding='utf8') as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"\nProcessing {file_path} ({total_lines} chunks)")

        pbar = tqdm(total=total_lines, desc=f"Encoding {os.path.basename(file_path)}", unit="chunk")

        for batch in read_jsonl_in_batches(file_path, batch_size=64):
            embedded = encode_chunks_with_metadata(batch, dense_encoder, sparse_encoder)
            save_chunks_to_chroma(embedded, chroma_collection, batch_size=64)
            pbar.update(len(batch))
        pbar.close()
    end_time = time.time()
    print(f"Total time used: {end_time - start_time}")