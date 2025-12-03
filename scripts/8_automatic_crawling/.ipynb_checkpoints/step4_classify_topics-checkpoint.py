#!/usr/bin/env python3
"""
STEP 4: Classify topics as politics/non-politics using ChatGPT API.

Usage: python step4_classify_topics.py --level 0

for testing: python step4_classify_topics.py --level 0 --dry-run

Output: classification/

MODIFICHE:
- Rimosso calcolo doc_topic_matrix, ora caricata da lda/doc_topic_matrix_level_{level}.npy
- Funzione extract_topic_data semplificata (usa matrice pre-calcolata)
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import requests

# ======================== CONFIGURATION ========================
def load_openai_key():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    secrets_path = os.path.join(BASE_DIR, "openai_secrets.json")
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r") as f:
                data = json.load(f)
            return data.get("OPENAI_API_KEY")
        except Exception as e:
            print(f"Warning: Could not read secrets file: {e}")
    return None

OPENAI_API_KEY = load_openai_key()
OPENAI_MODELS = [
    "gpt-5-nano",
    "gpt-5-nano-2025-08-07",
    "gpt-5-mini",
    "gpt-5-mini-2025-08-07"
]
API_URL = "https://api.openai.com/v1/chat/completions"
REQUEST_DELAY = 1.0
MAX_RETRIES = 3
SEED = 42
NUM_TOP_DOCS = 3
NUM_TOP_KEYWORDS = 40
NUM_RANDOM_DOCS = 3

# ======================== TIMING ========================
START_TIME = time.perf_counter()

def log_time(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[{elapsed:8.2f}s] {message}")

# ======================== CLASSIFICATION PROMPT ========================
CLASSIFICATION_PROMPT = """You are classifying topics from a topic model of Telegram messages.
Determine if this topic is related to POLITICS. Consider that the keywords are obtained with lda gensim.

POLITICS includes strictly: 
Let's consider as political all those topics that could be discussed during an electoral debate between candidates and presidents.


NON-POLITICS includes: cryptocurrency, finance, trading, entertainment, sports, gaming,
technology, science, personal topics, lifestyle, commercial content, religious content 
unless they are pretty political.

Topic keywords: {keywords}

Sample documents from this topic:
{documents}

Is this topic about POLITICS? Respond with ONLY one word: "yes" or "no"
"""

# ======================== LOAD MODEL AND DATA ========================
def load_lda_model_and_data(level: str):
    base_dir = f"../../results/levels_automatic/level_{level}"
    lda_dir = f"{base_dir}/lda"
    preprocess_dir = f"{base_dir}/preprocessing"

    best_k_path = f"{lda_dir}/best_k.json"
    if not os.path.exists(best_k_path):
        raise FileNotFoundError(f"best_k.json not found: {best_k_path}")
    with open(best_k_path, "r") as f:
        metadata = json.load(f)
    best_k = metadata["best_k"]

    model_path = f"{lda_dir}/models/lda_best.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LDA model not found: {model_path}")
    lda_model = joblib.load(model_path)

    doc_topic_matrix_path = f"{lda_dir}/doc_topic_matrix_level_{level}.npy"
    if not os.path.exists(doc_topic_matrix_path):
        raise FileNotFoundError(f"doc_topic_matrix not found: {doc_topic_matrix_path}")
    doc_topic_matrix = np.load(doc_topic_matrix_path)

    docs_path = f"{preprocess_dir}/messages_english_clean.tsv.gz"
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Documents not found: {docs_path}")
    df_docs = pd.read_csv(docs_path, sep="\t", compression="gzip", usecols=["text_lda", "text_llm"])
    df_docs = df_docs[df_docs["text_llm"].astype(str).str.strip() != ""]
    df_docs = df_docs[df_docs["text_lda"].astype(str).str.strip() != ""]
    docs_llm = df_docs["text_llm"].astype(str).tolist()

    return lda_model, doc_topic_matrix, docs_llm, best_k

def extract_topic_data(lda_model, doc_topic_matrix, docs_llm, topn=60):
    log_time("Using pre-computed doc_topic_matrix...")
    
    n_docs = doc_topic_matrix.shape[0]
    n_topics = doc_topic_matrix.shape[1]
    
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)

    rng = np.random.default_rng(SEED)

    topics_data = []
    for topic_idx in range(n_topics):
        keywords = [word for word, _ in lda_model.show_topic(topic_idx, topn=topn)]

        topic_scores = doc_topic_matrix[:, topic_idx]
        sorted_doc_indices = np.argsort(topic_scores)[::-1]
        top_doc_indices = sorted_doc_indices[:min(NUM_TOP_DOCS, n_docs)]

        top_docs = []
        for doc_idx in top_doc_indices:
            score = topic_scores[doc_idx]
            snippet = docs_llm[doc_idx][:300].replace("\n", " ")
            top_docs.append({"idx": int(doc_idx), "score": float(score), "text": snippet})

        candidates = np.where(dominant_topics == topic_idx)[0]
        candidates = np.setdiff1d(candidates, top_doc_indices)
        if candidates.size == 0:
            candidates = np.setdiff1d(np.arange(n_docs), top_doc_indices)

        random_docs = []
        if candidates.size > 0:
            n_sample = min(NUM_RANDOM_DOCS, candidates.size)
            random_indices = rng.choice(candidates, size=n_sample, replace=False)
            for doc_idx in random_indices:
                score = topic_scores[doc_idx]
                snippet = docs_llm[doc_idx][:300].replace("\n", " ")
                random_docs.append({"idx": int(doc_idx), "score": float(score), "text": snippet})

        topics_data.append({
            "topic_id": topic_idx,
            "keywords": keywords,
            "top_docs": top_docs,
            "random_docs": random_docs
        })

    return topics_data

# ======================== API FUNCTIONS ========================
def call_openai_api(prompt: str, model: str) -> str:
    if not OPENAI_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 700  # It is to be high beacuase they are reasoning models so they reason internally
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                error_detail = "Unknown"
                try:
                    error_detail = response.json().get("error", {}).get("message", "")[:150]
                except:
                    error_detail = response.text[:150]
                log_time(f"API error (attempt {attempt+1}): {response.status_code} - {error_detail}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                continue

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip().lower()
            return content

        except Exception as e:
            log_time(f"Request error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return None

def test_api_connection() -> str:
    if not OPENAI_API_KEY:
        return None

    log_time("Testing API connection...")
    for model in OPENAI_MODELS:
        log_time(f"  Trying model: {model}")
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with only: OK"}],
            "max_completion_tokens": 150
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                log_time(f"  ✓ Model {model} works!")
                return model
            else:
                error_detail = "Unknown"
                try:
                    error_detail = response.json().get("error", {}).get("message", "")[:100]
                except:
                    pass
                log_time(f"  ✗ Model {model} failed: {response.status_code} - {error_detail}")
        except Exception as e:
            log_time(f"  ✗ Model {model} error: {e}")

    return None

def format_documents_for_prompt(top_docs: list, random_docs: list) -> str:
    lines = []
    lines.append("Top 3 most representative documents:")
    for i, doc in enumerate(top_docs, 1):
        lines.append(f"  {i}) [score={doc['score']:.3f}] {doc['text'][:250]}...")

    lines.append("\n3 random documents from this topic:")
    for i, doc in enumerate(random_docs, 1):
        lines.append(f"  {i}) [score={doc['score']:.3f}] {doc['text'][:250]}...")

    return "\n".join(lines)

def classify_topic(topic_data: dict, model: str) -> bool:
    keywords_str = ", ".join(topic_data["keywords"][:NUM_TOP_KEYWORDS])
    documents_str = format_documents_for_prompt(topic_data["top_docs"], topic_data["random_docs"])

    prompt = CLASSIFICATION_PROMPT.format(keywords=keywords_str, documents=documents_str)

    response = call_openai_api(prompt, model)
    if response is None:
        log_time(f"    [DEBUG] API returned None")
        return None

    log_time(f"    [DEBUG] Raw response: '{response}'")
    
    response_clean = response.strip().lower()
    if response_clean in ["yes", "sì", "si", "y", "ok", "esatto", "exacly", ""]:
        return True
    elif response_clean in ["no", "n"]:
        return False
    else:
        if "yes" in response_clean:
            return True
        elif "no" in response_clean:
            return False
    
    log_time(f"    [DEBUG] Could not parse response")
    return None

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description="Classify topics using ChatGPT")
    parser.add_argument("--level", type=str, required=True, help="Hierarchy level")
    parser.add_argument("--dry-run", action="store_true", help="Don't call API, just show prompts")
    parser.add_argument("--test-api", action="store_true", help="Test API connection only")
    args = parser.parse_args()

    level = args.level
    log_time(f"Classifying topics for level {level}")

    # Paths
    base_dir = f"../../results/levels_automatic/level_{level}"
    classification_dir = f"{base_dir}/classification"
    os.makedirs(classification_dir, exist_ok=True)

    if not OPENAI_API_KEY:
        log_time("ERROR: No OpenAI API key found!")
        log_time("Set OPENAI_API_KEY environment variable or create openai_secrets.json")
        sys.exit(1)

    log_time(f"API key found: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}")

    working_model = test_api_connection()

    if args.test_api:
        if working_model:
            log_time(f"API test successful! Using model: {working_model}")
        else:
            log_time("API test FAILED - no working model found")
        return

    if not working_model:
        log_time("ERROR: Could not find a working model. Check your API key.")
        sys.exit(1)

    log_time(f"Using model: {working_model}")

    log_time("Loading LDA model and data...")
    try:
        lda_model, doc_topic_matrix, docs_llm, best_k = load_lda_model_and_data(level)
        log_time(f"Loaded model with {best_k} topics, {len(docs_llm)} documents")
        log_time(f"doc_topic_matrix shape: {doc_topic_matrix.shape}")
    except FileNotFoundError as e:
        log_time(f"ERROR: {e}")
        sys.exit(1)

    log_time("Extracting topic data (keywords + sample documents)...")
    topics_data = extract_topic_data(lda_model, doc_topic_matrix, docs_llm)
    log_time(f"Extracted data for {len(topics_data)} topics")

    # Output paths
    output_path = f"{classification_dir}/topics_classified.json"
    politics_topics_path = f"{classification_dir}/politics_topics.json"
    topics_json_path = f"{classification_dir}/topics_for_classification.json"

    with open(topics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "level": level,
            "num_topics": len(topics_data),
            "topics": topics_data
        }, f, indent=2, ensure_ascii=False)
    log_time(f"Saved topics data to {topics_json_path}")

    if args.dry_run:
        log_time("DRY RUN - showing first topic prompt:")
        topic = topics_data[0]
        keywords_str = ", ".join(topic["keywords"][:NUM_TOP_KEYWORDS])
        documents_str = format_documents_for_prompt(topic["top_docs"], topic["random_docs"])
        print(CLASSIFICATION_PROMPT.format(keywords=keywords_str, documents=documents_str))
        return

    num_topics = len(topics_data)
    politics_topics = []
    non_politics_topics = []
    error_topics = []
    classifications = []

    for i, topic in enumerate(topics_data):
        topic_id = topic["topic_id"]
        log_time(f"Classifying topic {topic_id}/{num_topics-1}...")

        result = classify_topic(topic, working_model)

        if result is True:
            politics_topics.append(topic_id)
            status = "POLITICS"
        elif result is False:
            non_politics_topics.append(topic_id)
            status = "non-politics"
        else:
            error_topics.append(topic_id)
            status = "ERROR"

        classifications.append({
            "topic_id": topic_id,
            "is_politics": result,
            "keywords": topic["keywords"][:10]
        })

        log_time(f"  -> {status}")

        if i < num_topics - 1:
            time.sleep(REQUEST_DELAY)

    output_data = {
        "level": level,
        "model_used": working_model,
        "num_topics": num_topics,
        "num_politics": len(politics_topics),
        "num_non_politics": len(non_politics_topics),
        "num_errors": len(error_topics),
        "politics_topics": politics_topics,
        "non_politics_topics": non_politics_topics,
        "error_topics": error_topics,
        "classifications": classifications
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log_time(f"Saved full results to {output_path}")

    with open(politics_topics_path, "w") as f:
        json.dump({
            "level": level,
            "politics_topics": politics_topics,
            "non_politics_topics": non_politics_topics
        }, f, indent=2)
    log_time(f"Saved politics topics list to {politics_topics_path}")

    log_time("=" * 50)
    log_time(f"SUMMARY:")
    log_time(f"  Total topics: {num_topics}")
    log_time(f"  Politics: {len(politics_topics)} ({100*len(politics_topics)/num_topics:.1f}%)")
    log_time(f"  Non-politics: {len(non_politics_topics)} ({100*len(non_politics_topics)/num_topics:.1f}%)")
    log_time(f"  Errors: {len(error_topics)}")
    log_time(f"  Politics topic IDs: {politics_topics}")

    total_time = time.perf_counter() - START_TIME
    log_time(f"COMPLETED in {total_time:.2f}s")

    with open(f"{classification_dir}/step4_completed.txt", "w") as f:
        f.write(f"Classification completed in {total_time:.2f}s\n")
        f.write(f"Model used: {working_model}\n")
        f.write(f"Politics topics: {len(politics_topics)}\n")
        f.write(f"Non-politics topics: {len(non_politics_topics)}\n")
        f.write(f"Errors: {len(error_topics)}\n")

if __name__ == "__main__":
    main()