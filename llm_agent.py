from openai import OpenAI
from dotenv import load_dotenv
import os
from retriever import load_encoder, load_collection, encode_query, retrieve_docs, query_rerank, expand_with_neighbors, dedup_by_chapter_event
from sentence_transformers import CrossEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# load llm api key in .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def build_rag_prompt(query, context):
    prompt = f"""已知资料如下：
{context}

用户提问：{query}
请参考所有已知资料, 并结合资料内容，简明、准确地回答问题。如果有多个符合的答案, 可以根据你是否确定而决定是否分别陈述这些答案.如果不能确定答案，请如实说明理由，不要凭空编造。"""
    return prompt

def llm_answer(query, expanded_results, model_name="gpt-4o"):
    context = expanded_results[0][0] if expanded_results else ""
    prompt = build_rag_prompt(query, context)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是BangDream知识问答助手, 也就是邦学家. 只能基于提供的资料内容作答。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    collection = load_collection()
    encoder = load_encoder()
    reranker = CrossEncoder("BAAI/bge-reranker-large")

    query_text = input("please enter your question：")
    print("Thinking...\n...")
    query_vec = encode_query(encoder, query_text)
    results = retrieve_docs(collection, query_vec, top_k=50)
    reranked = query_rerank(reranker, query_text, results, top_n=20)
    deduped = dedup_by_chapter_event(reranked, max_per_group=1)
    expanded_results = expand_with_neighbors(deduped[:5], collection)

    answer = llm_answer(query_text, expanded_results)

    print("\n=== Answer ===")
    print(answer)
    print("\n=== retrieved documents ===")
    for idx, (context, score, meta) in enumerate(expanded_results, 1):
        print(f"\n--- document {idx} (Score={score:.4f}) ---\n{context[:200]}...")
        print(meta)