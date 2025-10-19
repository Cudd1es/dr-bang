import json
import os
from glob import glob
import uuid

WINDOW_SIZE = 5
STEP = 2

def sliding_window_chunk(extracted_data, window_size=3, step=1):
    chunks = []
    n = len(extracted_data)
    for i in range(0, n - window_size + 1, step):
        chunk = extracted_data[i:i+window_size]
        chunks.append((i, i+window_size-1, chunk))
    return chunks

def process_story_file(file_path, story_type, window_size=3, step=1):
    with open(file_path, 'r', encoding='utf8') as f:
        story = json.load(f)
        if isinstance(story, list):
            chapters = story
        else:
            chapters = [story]
        result = []
        for chapter in chapters:
            data = chapter["extractedData"]
            event_name = chapter.get("eventName", "")
            chapter_title = chapter.get("chapterTitle", "")
            for start_idx, end_idx, chunk_list in sliding_window_chunk(data, window_size, step):
                result.append({
                    "text": chunk_list,
                    "eventName": event_name,
                    "chapterTitle": chapter_title,
                    "story_type": story_type,
                    "chunk_id": f"{uuid.uuid4().hex}",
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                })
        return result

def process_folder(folder_path, story_type, window_size=3, step=1, out_path="chunks.jsonl"):
    all_chunks = []
    files = glob(os.path.join(folder_path, '*.json'))
    for file_path in files:
        chunks = process_story_file(file_path, story_type, window_size, step)
        all_chunks.extend(chunks)

    with open(out_path, 'w', encoding='utf8') as fout:
        for chunk in all_chunks:
            fout.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"Saved {len(all_chunks)} chunks to {out_path}")

if __name__ == "__main__":
    process_folder('./stories/main_merged_stories', 'main', window_size=WINDOW_SIZE, step=STEP, out_path='./chunks/main_chunks.jsonl')
    process_folder('./stories/band_merged_stories', 'band', window_size=WINDOW_SIZE, step=STEP, out_path='./chunks/band_chunks.jsonl')
    process_folder('./stories/event_merged_stories', 'event', window_size=WINDOW_SIZE, step=STEP, out_path='./chunks/event_chunks.jsonl')
    process_folder('./stories/card_json_output', 'card', window_size=WINDOW_SIZE, step=STEP, out_path='./chunks/card_chunks.jsonl')