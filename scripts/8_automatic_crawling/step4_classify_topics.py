#!/usr/bin/env python3
"""
STEP 4: Classify topics as politics/non-politics using ChatGPT API.
Usage: python step4_classify_topics.py --level 0

This script sends each topic (keywords + sample docs) to ChatGPT
and asks whether it's related to politics.
"""

import os
import time
import argparse
import json
import requests

# ======================== CONFIGURATION ========================

def load_openai_key():
    secrets_path = os.path.join(os.path.dirname(__file__), "openai_secrets.json")
    try:
        with open(secrets_path, "r") as f:
            data = json.load(f)
        return data["OPENAI_API_KEY"]
    except FileNotFoundError:
        raise RuntimeError(
            "Missing OPENAI_API_KEY: set env var or create openai_secrets.json"
        )
    except KeyError:
        raise RuntimeError(
            "openai_secrets.json found but missing 'OPENAI_API_KEY' field"
        )

OPENAI_API_KEY = load_openai_key()
OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective for classification
API_URL = "https://api.openai.com/v1/chat/completions"

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3

# ======================== TIMING ========================
START_TIME = time.perf_counter()

def log_time(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[{elapsed:8.2f}s] {message}")

# ======================== CLASSIFICATION PROMPT ========================
CLASSIFICATION_PROMPT = """You are classifying topics from a topic model of Telegram messages.

Your task: Determine if this topic is related to POLITICS.

Definition of POLITICS includes:
- Elections, voting, political parties
- Politicians, government officials, political figures
- Government policies, laws, regulations
- Political movements, protests, activism
- International relations, diplomacy, wars
- Political ideologies (left/right, liberal/conservative)
- News about political events
- Conspiracy theories about governments or political figures

Definition of NON-POLITICS includes:
- Cryptocurrency, finance, trading (unless about regulations)
- Entertainment, sports, gaming
- Technology, science (unless about policy)
- Personal topics, lifestyle
- Commercial/advertising content
- Religious content (unless political)
- General news not about politics

Here is the topic:
Keywords: {keywords}

Respond with ONLY a JSON object in this exact format:
{{"is_politics": true/false, "confidence": "high"/"medium"/"low", "reason": "brief explanation"}}

Do not include any other text."""

# ======================== API FUNCTIONS ========================
def call_openai_api(prompt: str) -> dict:
    """Call OpenAI API and return parsed response."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # Low temperature for consistent classification
        "max_tokens": 200
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            return json.loads(content)
            
        except requests.exceptions.RequestException as e:
            log_time(f"API error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return {"is_politics": None, "confidence": "error", "reason": str(e)}
        
        except json.JSONDecodeError as e:
            log_time(f"JSON parse error: {content[:100]}...")
            return {"is_politics": None, "confidence": "error", "reason": f"JSON parse error: {e}"}

def classify_topic(topic_id: int, keywords: list) -> dict:
    """Classify a single topic."""
    keywords_str = ", ".join(keywords[:10])
    prompt = CLASSIFICATION_PROMPT.format(keywords=keywords_str)
    
    result = call_openai_api(prompt)
    result["topic_id"] = topic_id
    result["keywords"] = keywords[:10]
    
    return result

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description="Classify topics using ChatGPT")
    parser.add_argument("--level", type=str, required=True, help="Hierarchy level")
    parser.add_argument("--dry-run", action="store_true", help="Don't call API, just show prompts")
    args = parser.parse_args()
    
    level = args.level
    log_time(f"Classifying topics for level {level}")
    
    # Paths
    base_dir = f"../results/levels_automatic/level_{level}"
    lda_dir = f"{base_dir}/lda"
    
    topics_json_path = f"{lda_dir}/topics_for_classification.json"
    output_path = f"{lda_dir}/topics_classified.json"
    politics_topics_path = f"{lda_dir}/politics_topics.json"
    
    # Load topics
    if not os.path.exists(topics_json_path):
        log_time(f"ERROR: Topics file not found: {topics_json_path}")
        return
    
    with open(topics_json_path, "r") as f:
        topics_data = json.load(f)
    
    topics = topics_data["topics"]
    num_topics = len(topics)
    log_time(f"Loaded {num_topics} topics to classify")
    
    if args.dry_run:
        log_time("DRY RUN - showing first topic prompt:")
        keywords_str = ", ".join(topics[0]["keywords"][:10])
        print(CLASSIFICATION_PROMPT.format(keywords=keywords_str))
        return
    
    # Classify each topic
    classifications = []
    politics_topics = []
    non_politics_topics = []
    
    for i, topic in enumerate(topics):
        topic_id = topic["topic_id"]
        keywords = topic["keywords"]
        
        log_time(f"Classifying topic {topic_id}/{num_topics-1}...")
        
        result = classify_topic(topic_id, keywords)
        classifications.append(result)
        
        if result.get("is_politics") == True:
            politics_topics.append(topic_id)
            status = "POLITICS"
        elif result.get("is_politics") == False:
            non_politics_topics.append(topic_id)
            status = "non-politics"
        else:
            status = "ERROR"
        
        log_time(f"  -> {status} ({result.get('confidence', 'unknown')}): {result.get('reason', '')[:50]}")
        
        # Rate limiting
        if i < num_topics - 1:
            time.sleep(REQUEST_DELAY)
    
    # Save results
    output_data = {
        "level": level,
        "num_topics": num_topics,
        "num_politics": len(politics_topics),
        "num_non_politics": len(non_politics_topics),
        "politics_topics": politics_topics,
        "non_politics_topics": non_politics_topics,
        "classifications": classifications
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log_time(f"Saved full results to {output_path}")
    
    # Save simple list of politics topics
    with open(politics_topics_path, "w") as f:
        json.dump({
            "level": level,
            "politics_topics": politics_topics,
            "non_politics_topics": non_politics_topics
        }, f, indent=2)
    log_time(f"Saved politics topics list to {politics_topics_path}")
    
    # Summary
    log_time("=" * 50)
    log_time(f"SUMMARY:")
    log_time(f"  Total topics: {num_topics}")
    log_time(f"  Politics: {len(politics_topics)} ({100*len(politics_topics)/num_topics:.1f}%)")
    log_time(f"  Non-politics: {len(non_politics_topics)} ({100*len(non_politics_topics)/num_topics:.1f}%)")
    log_time(f"  Politics topic IDs: {politics_topics}")
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    log_time(f"COMPLETED in {total_time:.2f}s")
    
    with open(f"{lda_dir}/step4_completed.txt", "w") as f:
        f.write(f"Classification completed in {total_time:.2f}s\n")
        f.write(f"Politics topics: {len(politics_topics)}\n")
        f.write(f"Non-politics topics: {len(non_politics_topics)}\n")

if __name__ == "__main__":
    main()