from sentence_transformers import SentenceTransformer
from sentence_transformers.sparse_encoder import SparseEncoder
import torch
import os
import re

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
    for i, chunk in enumerate(chunks):
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

if __name__ == "__main__":
    chunks = [
        {"text": ["纱夜: 是吗。那告辞了",
                  "亚子: ……喂。燐燐，那、那个人拿着的是…… 吉他盒吧？ 她在玩乐队吧？真好啊，亚子也想组乐队……！",
                  "燐子: （小亚子经常会说起 自己组乐队的姐姐呢…… 乐队……对我来说是无法想象的世界……）",
                  "亚子: 燐燐，该走了！ 今天我有点想严格按照日程表来呢", "燐子: 啊。嗯……"], "eventName": "",
         "chapterTitle": "第1话: 绝不妥协的完美乐队", "story_type": "band",
         "chunk_id": "band_story_22_merged.json_26_30", "start_idx": 26, "end_idx": 30},
        {"text": ["有咲: （奥泽同学…… 加油……！！！）", "美咲: ……啊哈、哈哈哈…… ……也就是说，我想说的是，那个~~",
                  "美咲: 请多指教，或者应该说…… 今后也请多多关照……！", "美咲: 花、花咲川女子学园……学生会会长…… 奥泽美咲……",
                  "美咲: （哈啊~哈啊~~）"], "eventName": "Main Story", "chapterTitle": "第１话: 乘上春风",
         "story_type": "main", "chunk_id": "main_full.json_30_34", "start_idx": 30, "end_idx": 34}
    ]

    dense_encoder = DenseTextEncoder(DENSE_EMBEDDER_MODEL)
    sparse_encoder = SparseTextEncoder(SPARSE_EMBEDDER_MODEL)
    results = encode_chunks_with_metadata(chunks, dense_encoder, sparse_encoder)
    for entry in results:
        print(f"Chunk: {entry['chunk_id']}")
        print("Dense vector (shape):", entry['dense_embedding'].shape)
        print("Sparse vector (shape):", entry['sparse_embedding'].shape)
        print("Text:", entry['text'][:20], "...")
        print("Meta:", entry['eventName'], entry['chapterTitle'])
        print("chunk id:", entry['chunk_id'])
        print("---")


