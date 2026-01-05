#!/usr/bin/env python3
"""
STEP 4: Classify topics as political or non-political using ChatGPT.
Usage: python step4_classify_topics.py --level 0
       python step4_classify_topics.py --level 0 --base-dir ../../results/experiments/peak_jul_aug
"""

import os
import sys
import time
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

from openai import OpenAI

# ======================== TIMING ========================
START_TIME = time.perf_counter()
STEP_TIMES = {}

def log_time(msg: str) -> None:
    print(f"[{time.perf_counter() - START_TIME:8.2f}s] {msg}")

def start_timer(name: str) -> float:
    return time.perf_counter()

def end_timer(name: str, start: float) -> float:
    elapsed = time.perf_counter() - start
    STEP_TIMES[name] = elapsed
    return elapsed

# ======================== CONFIG ========================
NUM_TOP_KEYWORDS = 40
MODEL_NAME = "gpt-5-nano"  

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="../../results/levels_automatic",
                        help="Base directory for results")
    args = parser.parse_args()
    
    level = args.level
    base_dir = args.base_dir
    log_time(f"Classifying topics for level {level}")
    log_time(f"  Base dir: {base_dir}")
    
    # Paths
    level_dir = f"{base_dir}/level_{level}"
    topics_dir = f"{level_dir}/topics"
    classification_dir = f"{level_dir}/classification"
    os.makedirs(classification_dir, exist_ok=True)
    
    # Load API key
    t_start = start_timer("load_api_key")
    api_key_paths = [
        "../openai_secrets.json",
        "openai_secrets.json",
        os.path.expanduser("~/openai_secrets.json")
    ]
    
    api_key = None
    for path in api_key_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                secrets = json.load(f)
                api_key = secrets.get('api_key') or secrets.get('OPENAI_API_KEY')
                break
    
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        log_time("ERROR: OpenAI API key not found")
        sys.exit(1)
    
    log_time(f"API key found: {api_key[:10]}...{api_key[-4:]}")
    client = OpenAI(api_key=api_key)
    end_timer("load_api_key", t_start)
    
    # Test API
    t_start = start_timer("test_api")
    log_time("Testing API connection...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_completion_tokens=600
        )
        log_time(f"API test successful with model {MODEL_NAME}")
    except Exception as e:
        log_time(f"ERROR: API test failed: {e}")
        sys.exit(1)
    end_timer("test_api", t_start)
    
    # Load topics
    t_start = start_timer("load_topics")
    topics_file = f"{topics_dir}/topics_for_classification.json"
    if not os.path.exists(topics_file):
        log_time(f"ERROR: Topics file not found: {topics_file}")
        sys.exit(1)
    
    with open(topics_file, 'r') as f:
        topics_json = json.load(f)
    
    # Handle different formats
    if isinstance(topics_json, dict) and "topics" in topics_json:
        topics_data = topics_json["topics"]
    elif isinstance(topics_json, list):
        topics_data = topics_json
    else:
        topics_data = []
    
    log_time(f"Loaded {len(topics_data)} topics")
    end_timer("load_topics", t_start)
    
    # Classify topics
    t_start = start_timer("classify_topics")
    results = []
    politics_topics = []
    errors = 0
    
    for topic in topics_data:
        topic_id = topic['topic_id']
        keywords = topic.get('all_keywords', topic.get('keywords', []))[:NUM_TOP_KEYWORDS]
        
        log_time(f"Classifying topic {topic_id}/{len(topics_data)-1}...")
        
        prompt = f"""Analyze these keywords from a topic model and determine if this topic is about POLITICS.

Political topics include: elections, voting, political parties, government policies, politicians, legislation, political movements, campaigns, political ideologies, international relations, political news.

Keywords: {', '.join(keywords)}

Answer with ONLY 'yes' or 'no'."""
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=600
            )
            
            answer = response.choices[0].message.content.strip().lower()
            is_political = answer.startswith('yes') or answer == 'y'
            
            log_time(f"  -> {'POLITICS' if is_political else 'non-politics'} (raw: '{answer}')")
            
            results.append({
                "topic_id": topic_id,
                "keywords": keywords[:10],
                "is_political": is_political,
                "raw_response": answer
            })
            
            if is_political:
                politics_topics.append(topic_id)
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            log_time(f"  ERROR: {e}")
            errors += 1
            results.append({
                "topic_id": topic_id,
                "keywords": keywords[:10],
                "is_political": False,
                "error": str(e)
            })
    
    end_timer("classify_topics", t_start)
    
    # Save results
    t_start = start_timer("save_results")
    with open(f"{classification_dir}/topics_classified.json", 'w') as f:
        json.dump(results, f, indent=2)
    log_time(f"Saved full results to {classification_dir}/topics_classified.json")
    
    politics_data = {
        "politics_topics": politics_topics,
        "total_topics": len(topics_data),
        "political_count": len(politics_topics),
        "non_political_count": len(topics_data) - len(politics_topics)
    }
    
    with open(f"{classification_dir}/politics_topics.json", 'w') as f:
        json.dump(politics_data, f, indent=2)
    log_time(f"Saved politics topics list to {classification_dir}/politics_topics.json")
    end_timer("save_results", t_start)
    
    # Summary
    log_time("=" * 50)
    log_time("SUMMARY:")
    log_time(f"  Total topics: {len(topics_data)}")
    log_time(f"  Politics: {len(politics_topics)} ({100*len(politics_topics)/len(topics_data):.1f}%)")
    log_time(f"  Non-politics: {len(topics_data) - len(politics_topics)} ({100*(len(topics_data)-len(politics_topics))/len(topics_data):.1f}%)")
    log_time(f"  Errors: {errors}")
    log_time(f"  Politics topic IDs: {politics_topics}")
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    STEP_TIMES["total"] = total_time
    log_time(f"COMPLETED in {total_time:.2f}s")
    
    with open(f"{classification_dir}/step4_completed.txt", 'w') as f:
        f.write(f"Step 4: Topic Classification (ChatGPT)\n")
        f.write(f"Level: {level}\n")
        f.write(f"Base dir: {base_dir}\n")
        f.write(f"Status: COMPLETED\n")
        f.write(f"Total time: {total_time:.2f}s\n\n")
        f.write(f"Timing breakdown:\n")
        for step_name, step_time in STEP_TIMES.items():
            f.write(f"  {step_name}: {step_time:.2f}s\n")
        f.write(f"\nResults:\n")
        f.write(f"  Total topics: {len(topics_data)}\n")
        f.write(f"  Political topics: {len(politics_topics)}\n")
        f.write(f"  Non-political topics: {len(topics_data) - len(politics_topics)}\n")
        f.write(f"  Errors: {errors}\n")
        f.write(f"  Model: {MODEL_NAME}\n")

if __name__ == "__main__":
    main()