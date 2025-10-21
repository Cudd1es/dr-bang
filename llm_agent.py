from openai import OpenAI
from dotenv import load_dotenv
import os
from retriever import load_encoder, load_collection, encode_query, retrieve_docs, query_rerank, expand_with_neighbors, dedup_by_chapter_event
from sentence_transformers import CrossEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# load llm api key in .env
load_dotenv()


# openrouter method
DEEPSEEK_MODEL="deepseek/deepseek-chat-v3.1:free"
QWEN_MODEL="qwen/qwen3-235b-a22b:free"
#api_key = os.getenv("OPENROUTER_API_KEY")
#client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=api_key)

# open ai method
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
GPT_MODEL="gpt-4o"

def reformulate_query(user_question, model_name=GPT_MODEL):
    prompt = f"""你是一个BangDream知识检索助手。请把用户的问题扩写或转写为适合知识库语义检索的检索语句，涵盖所有可能的提问方式或同义关键词。
    用户问题：{user_question}
    """
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096,
    )
    return resp.choices[0].message.content.strip()

def build_rag_prompt(query, context):
    prompt = f"""你将获得多个独立的资料片段，请充分查阅每一条资料.
    已知资料如下：
{context}

用户提问：{query}
规则:
1. 请参考所有已知资料, 并结合资料内容，简明、准确地回答问题。
2. 如果有多个相关答案或不同观点，请全部分点列出并尽量注明出处.
3. 如果只能在部分资料里找到答案，也请说明是参考了哪几条资料.
4. 如果不能确定答案，请如实说明理由，不要凭空编造.
"""
    return prompt

def llm_answer(query, expanded_results, model_name=GPT_MODEL):
    context = expanded_results[0][0] if expanded_results else ""
    prompt = build_rag_prompt(query, context)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是BangDream知识问答助手, 也就是邦学家. 只能基于提供的资料内容作答。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=8192,
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    collection = load_collection()
    encoder = load_encoder()
    reranker = CrossEncoder("BAAI/bge-reranker-large")

    query_text = input("please enter your question：")
    # reformulate query
    print("Reformulating...")
    reformulated_query_text = reformulate_query(query_text)
    print(f"[DEBUG] reformulated query: {reformulated_query_text}")

    print("Thinking...\n...")
    # rerank original query
    query_vec = encode_query(encoder, query_text)
    results = retrieve_docs(collection, query_vec, top_k=20)
    reranked = query_rerank(reranker, query_text, results, top_n=10)

    # rerank reformulated query
    reformulated_query_vec = encode_query(encoder, reformulated_query_text)
    reformulated_results = retrieve_docs(collection, reformulated_query_vec, top_k=20)
    reformulated_reranked = query_rerank(reranker, reformulated_query_text, reformulated_results, top_n=10)

    total_reranked = reranked + reformulated_reranked
    deduped = dedup_by_chapter_event(total_reranked, max_per_group=1)
    expanded_results = expand_with_neighbors(deduped[:5], collection)

    answer = llm_answer(query_text, expanded_results)

    print("\n=== Answer ===")
    print(answer)
    print("\n=== retrieved documents ===")
    for idx, (context, score, meta) in enumerate(expanded_results, 1):
        print(f"\n--- document {idx} (Score={score:.4f}) ---\n{context[:200]}...")
        print(meta)