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

if __name__ == "__main__":
    collection = load_collection()
    encoder = load_encoder()

    query_text = "乐奈喜欢什么?"
    query_vec = encode_query(encoder, query_text)
    results = retrieve_docs(collection, query_vec, top_k=50)
    reranked = query_rerank(reranker, query_text, results, top_n=5)
