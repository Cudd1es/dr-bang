import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "bangdream_dense"
MODEL_NAME = "BAAI/bge-base-zh-v1.5"

reranker = CrossEncoder("BAAI/bge-reranker-large")


def load_collection(db_path=CHROMA_DB_DIR, collection_name=COLLECTION_NAME):
    """Connect to Chroma persistent DB and load a collection."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)
    return collection

def load_encoder(model_name=MODEL_NAME):
    """Load dense encoder model."""
    return SentenceTransformer(model_name)

def encode_query(encoder, query_text):
    """Encode query text into normalized embedding."""
    return encoder.encode_query([query_text], normalize_embeddings=True)

def retrieve_docs(collection, query_vec, top_k=5):
    """Retrieve documents from Chroma collection."""
    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    return results

def query_rerank(reranker, query, results, top_n=3):
    """Use CrossEncoder to re-rank retrieved results."""
    docs = results["documents"][0]
    pairs = [(query, doc) for doc in docs]

    # CrossEncoder
    scores = reranker.predict(pairs)

    # rerank
    ranked = sorted(zip(docs, scores, results["metadatas"][0]), key=lambda x: x[1], reverse=True)

    # get top_n
    reranked_docs = ranked[:top_n]

    print("=== After Rerank ===")
    for i, (doc, score, meta) in enumerate(reranked_docs, 1):
        print(f"Rank {i} | Score: {score:.4f}")
        print(meta)
        print(doc)
        print("-" * 40)

    return reranked_docs

def pretty_print_results(results):
    """Nicely print retrieved results."""
    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]
    for idx, (doc, dist, meta) in enumerate(zip(docs, dists, metas)):
        print(f"Rank {idx + 1} | Distance: {dist:.4f}")
        print(meta)
        print(doc)
        print("-" * 40)

# expend documents
def get_all_chunks_in_chapter(collection, chapter_title, event_name=None, story_type=None):
    filters = []
    if chapter_title:
        filters.append({"chapterTitle": chapter_title})
    if story_type:
        filters.append({"story_type": story_type})
    if event_name:
        filters.append({"eventName": event_name})
    if len(filters) == 1:
        filter_dict = filters[0]
    elif len(filters) > 1:
        filter_dict = {"$and": filters}
    else:
        filter_dict = {}
    results = collection.get(where=filter_dict, include=["documents", "metadatas"])
    chunk_list = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        chunk_list.append({
            "text": doc,
            **meta,
        })
    return chunk_list

def find_adjacent_chunks(current_chunk, all_chunks):
    start_idx = current_chunk['start_idx']
    end_idx = current_chunk['end_idx']
    prev_chunk, next_chunk = None, None
    for chunk in all_chunks:
        if chunk['end_idx'] == start_idx - 1:
            prev_chunk = chunk
        if chunk['start_idx'] == end_idx + 1:
            next_chunk = chunk
    return prev_chunk, next_chunk

def safe_to_list(x):
    if isinstance(x, str):
        return x.split('\n') if '\n' in x else [x]
    return list(x)

def expand_with_neighbors(reranked_docs, collection):
    expanded_results = []
    for doc, score, meta in reranked_docs:
        print(meta)
        chapter_title = meta.get("chapterTitle", "")
        event_name = meta.get("eventName", "")
        story_type = meta.get("story_type", None)
        all_chunks = get_all_chunks_in_chapter(collection, chapter_title, event_name, story_type)
        prev_chunk, next_chunk = find_adjacent_chunks(meta, all_chunks)
        expanded_text = []
        if prev_chunk:
            #expanded_text += prev_chunk["text"]
            expanded_text += safe_to_list(prev_chunk["text"])
            #expanded_text.extend(prev_chunk["text"])
        #expanded_text += doc
        expanded_text += safe_to_list(doc)

        #expanded_text.extend(doc if isinstance(doc, list) else [doc])
        if next_chunk:
            #expanded_text.extend(next_chunk["text"])
            #expanded_text += next_chunk["text"]
            expanded_text += safe_to_list(next_chunk["text"])

        expanded_results.append((
            "\n".join(expanded_text),
            score,
            {
                **meta,
                #"prev_chunk_id": prev_chunk["ids"][0] if prev_chunk else None,
                #"next_chunk_id": next_chunk["ids"][0] if next_chunk else None,
            }
        ))
    return expanded_results

if __name__ == "__main__":
    collection = load_collection()
    encoder = load_encoder()

    query_text = "乐奈喜欢什么?"
    query_vec = encode_query(encoder, query_text)
    results = retrieve_docs(collection, query_vec, top_k=20)
    reranked = query_rerank(reranker, query_text, results, top_n=5)

    expanded_results = expand_with_neighbors(reranked, collection)
    for doc in expanded_results:
        print("===")
        print(doc)
        print(doc[0])
        print("===")
