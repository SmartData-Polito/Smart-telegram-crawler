#!/usr/bin/env python3
"""
STEP 4: Classify topics as GAMING or NON-GAMING using ChatGPT.
Usa keywords + TOP 3 documenti + 3 documenti RANDOM per ogni topic.

Usage: python step4_classify_topics.py --level 0 --base-dir ../../results/experiments_tgdataset/threshold_40_pure
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
    parser.add_argument("--base-dir", type=str, required=True,
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
    gaming_topics = []
    errors = 0
    
    for topic in topics_data:
        topic_id = topic['topic_id']
        keywords = topic.get('all_keywords', topic.get('keywords', []))[:NUM_TOP_KEYWORDS]
        
        # Get top_documents and random_documents
        top_docs = topic.get('top_documents', [])
        random_docs = topic.get('random_documents', [])
        
        # Fallback to sample_documents if new fields not present
        if not top_docs and not random_docs:
            sample_docs = topic.get('sample_documents', [])
            top_docs = sample_docs[:3] if len(sample_docs) >= 3 else sample_docs
            random_docs = sample_docs[3:6] if len(sample_docs) > 3 else []
        
        log_time(f"Classifying topic {topic_id}/{len(topics_data)-1}...")
        
        # Build prompt with keywords + documents
        prompt = f"""Analyze this topic from a topic model and determine if it is about GAMING.

GAMING topics include:
- Videogame modding
- Video games (any platform: PC, console, mobile)
- Game mods, cheats, hacks, aimbots
- Esports, gaming tournaments
- Specific games (Minecraft, Fortnite, PUBG, COD, GTA, etc.)
- Gaming communities, clans, guilds
- Game streaming (Twitch, YouTube Gaming)

NOT GAMING:
- General entertainment (movies, TV, music): NO
- General software/tech: NO
- Crypto/trading/NFT: NO
- Adult content: NO
- News, politics: NO
- Social media, memes (unless specifically gaming memes): NO
- Real warfare, military news, geopolitical conflicts: NO
- Airsoft, paintball, real weapons: NO
- Gambling, betting, casinos (unless video game gambling): NO
- Sports betting, fantasy sports: NO
- Card games (poker, blackjack - unless digital/video game): NO
- Board games (unless digital versions): NO
- Scams, money doubling, carding, fraud: NO

=== KEYWORDS ===
{', '.join(keywords)}

=== TOP 3 REPRESENTATIVE MESSAGES ===
"""
        
        # Add top documents
        for i, doc in enumerate(top_docs[:3], 1):
            prompt += f"{i}. {doc}\n"
        
        if not top_docs:
            prompt += "(no top documents available)\n"
        
        prompt += "\n=== 3 RANDOM MESSAGES ===\n"
        
        # Add random documents
        for i, doc in enumerate(random_docs[:3], 1):
            prompt += f"{i}. {doc}\n"
        
        if not random_docs:
            prompt += "(no random documents available)\n"
        
        prompt += "\nBased on the keywords and messages above, is this topic about GAMING?\nAnswer with ONLY 'yes' or 'no'."
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=600
            )
            
            answer = response.choices[0].message.content.strip().lower()
            is_gaming = answer.startswith('yes') or answer == 'y'
            
            log_time(f"  -> {'GAMING' if is_gaming else 'non-gaming'} (raw: '{answer}')")
            
            results.append({
                "topic_id": topic_id,
                "keywords": keywords[:10],
                "top_documents": top_docs[:3],
                "random_documents": random_docs[:3],
                "is_gaming": is_gaming,
                "raw_response": answer
            })
            
            if is_gaming:
                gaming_topics.append(topic_id)
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            log_time(f"  ERROR: {e}")
            errors += 1
            results.append({
                "topic_id": topic_id,
                "keywords": keywords[:10],
                "is_gaming": False,
                "error": str(e)
            })
    
    end_timer("classify_topics", t_start)
    
    # Save results
    t_start = start_timer("save_results")
    
    # Full results
    with open(f"{classification_dir}/topics_classified.json", 'w') as f:
        json.dump(results, f, indent=2)
    log_time(f"Saved full results to {classification_dir}/topics_classified.json")
    
    # Gaming topics summary
    gaming_data = {
        "gaming_topics": gaming_topics,
        "total_topics": len(topics_data),
        "gaming_count": len(gaming_topics),
        "non_gaming_count": len(topics_data) - len(gaming_topics)
    }
    
    with open(f"{classification_dir}/gaming_topics.json", 'w') as f:
        json.dump(gaming_data, f, indent=2)
    log_time(f"Saved gaming topics list to {classification_dir}/gaming_topics.json")
    
    # Also save as politics_topics.json for compatibility with step5
    compat_data = {
        "politics_topics": gaming_topics,  # Compatibility key
        "gaming_topics": gaming_topics,
        "total_topics": len(topics_data),
        "political_count": len(gaming_topics),
        "non_political_count": len(topics_data) - len(gaming_topics)
    }
    
    with open(f"{classification_dir}/politics_topics.json", 'w') as f:
        json.dump(compat_data, f, indent=2)
    log_time(f"Saved compatibility file to {classification_dir}/politics_topics.json")
    
    end_timer("save_results", t_start)
    
    # Summary
    log_time("=" * 50)
    log_time("SUMMARY:")
    log_time(f"  Total topics: {len(topics_data)}")
    log_time(f"  Gaming: {len(gaming_topics)} ({100*len(gaming_topics)/len(topics_data):.1f}%)")
    log_time(f"  Non-gaming: {len(topics_data) - len(gaming_topics)} ({100*(len(topics_data)-len(gaming_topics))/len(topics_data):.1f}%)")
    log_time(f"  Errors: {errors}")
    log_time(f"  Gaming topic IDs: {gaming_topics}")
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    STEP_TIMES["total"] = total_time
    log_time(f"COMPLETED in {total_time:.2f}s")
    
    with open(f"{classification_dir}/step4_completed.txt", 'w') as f:
        f.write(f"Step 4: Topic Classification (Gaming - ChatGPT)\n")
        f.write(f"Level: {level}\n")
        f.write(f"Base dir: {base_dir}\n")
        f.write(f"Status: COMPLETED\n")
        f.write(f"Total time: {total_time:.2f}s\n\n")
        f.write(f"Timing breakdown:\n")
        for step_name, step_time in STEP_TIMES.items():
            f.write(f"  {step_name}: {step_time:.2f}s\n")
        f.write(f"\nResults:\n")
        f.write(f"  Total topics: {len(topics_data)}\n")
        f.write(f"  Gaming topics: {len(gaming_topics)}\n")
        f.write(f"  Non-gaming topics: {len(topics_data) - len(gaming_topics)}\n")
        f.write(f"  Errors: {errors}\n")
        f.write(f"  Model: {MODEL_NAME}\n")

if __name__ == "__main__":
    main()