from sentence_transformers import SentenceTransformer
from sentence_transformers.sparse_encoder import SparseEncoder
import torch
import os
import re
from tqdm import tqdm
import numpy as np
import chromadb
import json
from opensearchpy import OpenSearch
import time


DENSE_EMBEDDER_MODEL = "BAAI/bge-base-zh-v1.5"
SPARSE_EMBEDDER_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"


class TextCleaner:
    def __init__(self, lowercase=True, remove_urls=True, normalize_space=True):
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

    def encode(self, texts):
        if isinstance(texts, str) or isinstance(texts, list) and isinstance(texts[0], str):
            texts = [join_chunk_text(t) if isinstance(t, list) else t for t in (texts if isinstance(texts, list) else [texts])]
        else:
            raise ValueError("Input should be str or list of str/list.")

        cleaned = [self.cleaner.clean(t) for t in texts]
        output = self.model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=self.normalize)
        return output


class SparseTextEncoder:
    """
    output: torch tensor
    """
    def __init__(self, model_name, device=None):
        self.device = device or "cpu"
        self.encoder = SparseEncoder(model_name, device=self.device)
        self.cleaner = TextCleaner()

    def encode(self, texts):
        if isinstance(texts, str) or isinstance(texts, list) and isinstance(texts[0], str):
            texts = [join_chunk_text(t) if isinstance(t, list) else t for t in
                     (texts if isinstance(texts, list) else [texts])]
        else:
            raise ValueError("Input should be str or list of str/list.")
        cleaned = [self.cleaner.clean(t) for t in texts]
        output = self.encoder.encode(cleaned)
        return output

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
      "chunk_id": "event_2_1_0_2",
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
    dense_vecs = dense_encoder.encode(text)
    sparse_vecs = sparse_encoder.encode(text)
    result = []
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Encoding Chunks"):
        result.append({
            "chunk_id": chunk.get("chunk_id"),
            "dense_embedding": dense_vecs[i],
            "sparse_embedding": sparse_vecs[i],
            "text": text[i],
            "eventName": chunk.get("eventName"),
            "chapterTitle": chunk.get("chapterTitle"),
            "story_type": chunk.get("story_type"),
            "start_idx": chunk.get("start_idx"),
            "end_idx": chunk.get("end_idx"),
        })
    return result

# tmp
def encode_chunks_with_metadata_2(chunks, dense_encoder, sparse_encoder=None):
    texts = [join_chunk_text(chunk["text"]) for chunk in chunks]
    dense_vecs = dense_encoder.encode(texts)
    if sparse_encoder is not None:
        sparse_vecs = sparse_encoder.encode(texts)
    result = []
    for i, chunk in enumerate(chunks):
        entry = {
            "chunk_id": chunk.get("chunk_id"),
            "dense_embedding": dense_vecs[i],
            "text": texts[i],
            "eventName": chunk.get("eventName"),
            "chapterTitle": chunk.get("chapterTitle"),
            "story_type": chunk.get("story_type"),
            "start_idx": chunk.get("start_idx"),
            "end_idx": chunk.get("end_idx"),
        }
        if sparse_encoder is not None:
            entry["sparse_embedding"] = sparse_vecs[i]
        result.append(entry)
    return result
#!tmp


# save dense embedding to chroma vector database
def save_chunks_to_chroma(embedded_chunk, chroma_db_dir="./chroma_db", collection_name="bangdream"):
    client = chromadb.PersistentClient(path=chroma_db_dir)
    collection = client.get_or_create_collection(collection_name)
    ids = []
    documents = []
    embeddings = []
    metadata = []

    for entry in embedded_chunk:
        ids.append(entry["chunk_id"])
        documents.append(entry["text"])
        embeddings.append(entry["dense_embedding"].tolist() if isinstance(entry["dense_embedding"], np.ndarray) else entry["dense_embedding"])
        meta = {k: v for k, v in entry.items() if k not in ["chunk_id", "dense_embedding", "sparse_embedding", "text"]}
        metadata.append(meta)

    batch_size = 64
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadata[i:i + batch_size]
        )
    print(f"saved {len(ids)} chunks to {collection_name}")

# save sparse embedding to OpenSearch
def store_sparse_chunk(index_name, chunk, client):
    sparse_tensor = chunk["sparse_embedding"].coalesce()
    indices = sparse_tensor.indices().squeeze().tolist()
    values = sparse_tensor.values().tolist()
    doc = {
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "eventName": chunk.get("eventName"),
        "chapterTitle": chunk.get("chapterTitle"),
        "story_type": chunk.get("story_type"),
        "start_idx": chunk.get("start_idx"),
        "end_idx": chunk.get("end_idx"),
        "sparse_embedding": {
            "indices": indices,
            "values": values
        }
    }
    resp = client.index(index=index_name, body=doc)
    return resp

def bulk_store_sparse_chunks(index_name, chunk_list, client):
    for chunk in chunk_list:
        store_sparse_chunk(index_name, chunk, client)
    print(f"Saved {len(chunk_list)} sparse chunks to {index_name}")

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

    chunks = [
        {"text": ["莉莎: 友希那☆ 我现在准备去新开的饰品店。 友希那也一起……",
                  "友希那: 我没兴趣……今天入场时间早。我很赶时间", "莉莎: 等、等一下啦！ ……那至少陪我走一段……啊！",
                  "其他班级的学生A: 对不起，撞到你了，没事吧？", "莉莎: 我才是不好意思！都怪我东张西望的……"],
         "eventName": "", "chapterTitle": "第1话: 绝不妥协的完美乐队", "story_type": "band",
         "chunk_id": "9b28e9938292486e9a61f2d1787bb828", "start_idx": 0, "end_idx": 4},
        {"text": ["LAYER: 嗯，志愿调查表。 老师说我们已经三年级了，所以要填一下",
                  "PAREO: PAREO这边好像也快发了…… 这么说来，我们都已经是三年级生了呢",
                  "LAYER: 只不过一个是高中生一个是初中生", "PAREO: LAYER同学，您已经决定好毕业要做什么了吗？",
                  "LAYER: 嗯~你要看我的志愿调查表吗？"], "eventName": "Main Story", "chapterTitle": "第２０话: 扣动扳机",
         "story_type": "main", "chunk_id": "521f6b7e92c140298dece700959e46dc", "start_idx": 2, "end_idx": 6}

    ]


    results = encode_chunks_with_metadata(chunks, dense_encoder, sparse_encoder)
    for entry in results:
        print(f"Chunk: {entry['chunk_id']}")
        print("Dense vector (shape):", entry['dense_embedding'].shape)
        print("Sparse vector (shape):", entry['sparse_embedding'].shape)
        print("Text:", entry['text'][:20], "...")
        print("Meta:", entry['eventName'], entry['chapterTitle'])
        print("chunk id:", entry['chunk_id'])
        print("---")



    print(results[0])
    
    
    # connect to OpenSearch
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True
    )
    bulk_store_sparse_chunks("bangdream_sparse", results, client)

"""
    # save to chroma
    start_time = time.time()
    for file_path in chunk_files:
        for batch in tqdm(read_jsonl_in_batches(file_path, batch_size=64), desc=f"Encoding {file_path}"):
            # 只需要dense，可以让sparse_encoder=None并略过sparse_embedding
            embedded = encode_chunks_with_metadata_2(batch, dense_encoder, sparse_encoder=None)
            save_chunks_to_chroma(embedded)
    end_time = time.time()
    print(f"time used: {end_time - start_time}")

"""