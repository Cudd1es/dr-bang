# Dr-Bang: BangDream! Knowledge RAG System

A local, extensible Retrieval-Augmented Generation (RAG) system for **BangDream!** knowledge Q&A.  
Supports dense retrieval, cross-encoder rerank, and LLM answer generation with full context expansion.

---

## Features

- **Chunking**: Splits all story files (main/band/event/card) into overlapping semantic chunks with metadata.
- **Embedding**: Encodes chunks using `BAAI/bge-base-zh-v1.5` dense embeddings; stores in local ChromaDB.
- **Retrieving**: Retrieves top-k relevant chunks, re-ranks with `bge-reranker-large`, deduplicates by chapter/event.
- **RAG Agent**: Reformulates queries with LLM, runs query retrieval, context expansion, and generates final answer via LLM.

---

## Structure

```
chunker.py         # Story chunking (window/step, per type) & batch export
embedder.py        # Embedding & storing chunks in ChromaDB
retriever.py       # Dense retrieval, rerank, context expansion
llm_agent.py       # Main QA agent (retrieval + LLM answer generation)
requirements.txt   # All Python dependencies
.env               # Your API keys
/chunks/           # Generated chunk files (jsonl)
/chroma_db/        # Local ChromaDB persistent storage
/stories/          # All input story JSONs (merged by type)
```

---

## Installation

1. **Clone this repo**
    ```bash
    git clone https://github.com/Cudd1es/dr-bang.git
    cd dr-bang
    ```

2. **Python environment (Python 3.9+)**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare input stories**
    - Place all story JSON files (main/band/event/card) in `./stories/` as described above.

4. **Set your API Key**
    - Create a `.env` file in the project root:
      ```
      OPENAI_API_KEY=sk-your-key-here
      ```
    - _(Optional: for OpenRouter, use `OPENROUTER_API_KEY` instead.)_

---

## Usage

1. **Chunk your stories**  
   Generate chunked `.jsonl` files (per story type):
   ```bash
   python chunker.py
   ```

2. **Encode and store chunks**
   ```bash
   python embedder.py
   ```

3. **Run RAG QA agent**
   ```bash
   python llm_agent.py
   ```

   Enter your BangDream! related question at the prompt.  
   The agent will:
   - Reformulate your query
   - Retrieve and rerank relevant story chunks
   - Expand context with adjacent text
   - Call LLM to generate a grounded answer
   - Print supporting evidence

---

## Model Notes

- **Dense Encoder**: [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)
- **Cross-Encoder Reranker**: [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)
- **LLM Answering**: GPT-4o (via OpenAI API) or OpenRouter compatible models  
  (You can swap to DeepSeek/Qwen/Qianfan/etc. by editing `llm_agent.py`.)

---

## Customization & Extensibility & Todo

- Add documents of general BangDream! background as LLM reference.
- To add new story types or other sources, place/merge files in `stories/` and update `chunker.py`.
- For sparse retrieval, extend `embedder.py`/`retriever.py` as needed (placeholders included).

---

## Requirements

- Python 3.9+
- `sentence-transformers`, `chromadb`, `tqdm`, `openai`, `python-dotenv`  
  _(see `requirements.txt` for full list)_

---

## License

Apache 2.0 License.  
This project is not affiliated with or endorsed by Bushiroad or the creators of BanG Dream!.

---

## Acknowledgements

- [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI GPT-4o](https://platform.openai.com/)
- [Bestdori](https://bestdori.com)
