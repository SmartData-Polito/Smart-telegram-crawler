#!/usr/bin/env python3
"""
Estrae tutti i dati interessanti per la tesi da ogni esperimento.
Usage: python extract_thesis_data.py > thesis_data.txt
"""

import json
import os
import numpy as np
import pandas as pd
from glob import glob

def print_section(title):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

def print_subsection(title):
    print(f"\n{'-'*60}")
    print(f" {title}")
    print(f"{'-'*60}")

def analyze_level_detailed(exp_name, level, threshold):
    """Analizza un singolo livello in dettaglio."""
    
    base_dir = f"../../results/experiments/{exp_name}/level_{level}"
    
    # Check files exist
    preprocess_file = f"{base_dir}/preprocessing/messages_english_clean.tsv.gz"
    matrix_file = f"{base_dir}/lda/doc_topic_matrix_level_{level}.npy"
    politics_file = f"{base_dir}/classification/politics_topics.json"
    tracking_file = f"{base_dir}/preprocessing/channels_tracking.json"
    best_k_file = f"{base_dir}/lda/best_k.json"
    topics_file = f"{base_dir}/topics/topics_for_classification.json"
    
    if not os.path.exists(preprocess_file):
        return None
    if not os.path.exists(matrix_file):
        return None
    
    result = {"level": level, "threshold": threshold}
    
    # === PREPROCESSING DATA ===
    if os.path.exists(tracking_file):
        with open(tracking_file) as f:
            tracking = json.load(f)
        result["preprocessing"] = {
            "total_nodes_input": tracking.get("total_nodes", 0),
            "total_messages_before_filter": tracking.get("total_messages_before_date_filter", 0),
            "total_messages_after_filter": tracking.get("total_messages", 0),
            "english_messages": tracking.get("english_messages", 0),
            "channels_no_folder": tracking.get("summary", {}).get("channels_no_folder", 0),
            "channels_no_files": tracking.get("summary", {}).get("channels_no_files", 0),
            "channels_no_english": tracking.get("summary", {}).get("channels_no_english", 0),
            "channels_only_short": tracking.get("summary", {}).get("channels_only_short_messages", 0),
            "channels_final": tracking.get("summary", {}).get("channels_with_valid_messages", 0),
            "date_filter_start": tracking.get("start_date"),
            "date_filter_end": tracking.get("end_date")
        }
    
    # === LDA DATA ===
    if os.path.exists(best_k_file):
        with open(best_k_file) as f:
            lda_params = json.load(f)
        result["lda"] = {
            "num_topics": lda_params.get("best_k", 0),
            "coherence": lda_params.get("best_coherence", 0),
            "vocab_size": lda_params.get("vocab_size", 0),
            "num_documents": lda_params.get("n_documents", 0),
            "k_range": lda_params.get("k_range", [0, 0])
        }
    
    # === CLASSIFICATION DATA ===
    if os.path.exists(politics_file):
        with open(politics_file) as f:
            politics_data = json.load(f)
        
        pol_topics = set(politics_data.get("politics_topics", []))
        total_topics = politics_data.get("total_topics", 0)
        
        result["classification"] = {
            "total_topics": total_topics,
            "political_topics": len(pol_topics),
            "non_political_topics": total_topics - len(pol_topics),
            "pct_political_topics": 100 * len(pol_topics) / total_topics if total_topics > 0 else 0
        }
    else:
        pol_topics = set()
    
    # === LOAD MESSAGES AND MATRIX ===
    try:
        df = pd.read_csv(preprocess_file, sep='\t', compression='gzip')
        doc_topic_matrix = np.load(matrix_file)
    except Exception as e:
        print(f"    Errore caricamento dati: {e}")
        return result
    
    # === MESSAGE-LEVEL ANALYSIS ===
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    df['dominant_topic'] = dominant_topics
    df['is_political_message'] = df['dominant_topic'].isin(pol_topics)
    
    total_messages = len(df)
    political_messages = int(df['is_political_message'].sum())
    
    result["messages"] = {
        "total": total_messages,
        "political": political_messages,
        "non_political": total_messages - political_messages,
        "pct_political": 100 * political_messages / total_messages if total_messages > 0 else 0
    }
    
    # === CHANNEL-LEVEL ANALYSIS ===
    channel_stats = df.groupby('channel_id').agg(
        total_messages=('is_political_message', 'count'),
        political_messages=('is_political_message', 'sum')
    ).reset_index()
    
    channel_stats['non_political_messages'] = channel_stats['total_messages'] - channel_stats['political_messages']
    channel_stats['political_ratio'] = channel_stats['political_messages'] / channel_stats['total_messages']
    channel_stats['is_political_channel'] = channel_stats['political_ratio'] >= threshold
    
    total_channels = len(channel_stats)
    political_channels = int(channel_stats['is_political_channel'].sum())
    
    result["channels"] = {
        "total": total_channels,
        "political": political_channels,
        "non_political": total_channels - political_channels,
        "pct_political": 100 * political_channels / total_channels if total_channels > 0 else 0
    }
    
    # === POLITICAL CHANNELS METRICS ===
    df_pol = channel_stats[channel_stats['is_political_channel']]
    if len(df_pol) > 0:
        result["political_channels_metrics"] = {
            "count": len(df_pol),
            "avg_messages": round(df_pol['total_messages'].mean(), 2),
            "avg_political_messages": round(df_pol['political_messages'].mean(), 2),
            "avg_non_political_messages": round(df_pol['non_political_messages'].mean(), 2),
            "median_messages": round(df_pol['total_messages'].median(), 2),
            "min_messages": int(df_pol['total_messages'].min()),
            "max_messages": int(df_pol['total_messages'].max()),
            "avg_political_ratio": round(df_pol['political_ratio'].mean() * 100, 2)
        }
    
    # === NON-POLITICAL CHANNELS METRICS ===
    df_non_pol = channel_stats[~channel_stats['is_political_channel']]
    if len(df_non_pol) > 0:
        result["non_political_channels_metrics"] = {
            "count": len(df_non_pol),
            "avg_messages": round(df_non_pol['total_messages'].mean(), 2),
            "avg_political_messages": round(df_non_pol['political_messages'].mean(), 2),
            "avg_non_political_messages": round(df_non_pol['non_political_messages'].mean(), 2),
            "median_messages": round(df_non_pol['total_messages'].median(), 2),
            "min_messages": int(df_non_pol['total_messages'].min()),
            "max_messages": int(df_non_pol['total_messages'].max()),
            "avg_political_ratio": round(df_non_pol['political_ratio'].mean() * 100, 2)
        }
    
    # === TOPIC-LEVEL ANALYSIS ===
    topic_stats = []
    for topic_id in range(doc_topic_matrix.shape[1]):
        topic_mask = df['dominant_topic'] == topic_id
        topic_messages = int(topic_mask.sum())
        is_political = topic_id in pol_topics
        
        topic_stats.append({
            "topic_id": topic_id,
            "is_political": is_political,
            "messages": topic_messages
        })
    
    topic_stats = sorted(topic_stats, key=lambda x: x['messages'], reverse=True)
    
    pol_topic_msgs = [t['messages'] for t in topic_stats if t['is_political']]
    non_pol_topic_msgs = [t['messages'] for t in topic_stats if not t['is_political']]
    
    result["topic_metrics"] = {
        "political_topics_total_messages": sum(pol_topic_msgs),
        "non_political_topics_total_messages": sum(non_pol_topic_msgs),
        "avg_messages_per_political_topic": round(np.mean(pol_topic_msgs), 2) if pol_topic_msgs else 0,
        "avg_messages_per_non_political_topic": round(np.mean(non_pol_topic_msgs), 2) if non_pol_topic_msgs else 0,
        "top_5_topics": topic_stats[:5]
    }
    
    # === LOAD TOPIC KEYWORDS ===
    if os.path.exists(topics_file):
        with open(topics_file) as f:
            topics_data = json.load(f)
        
        keywords_map = {}
        if isinstance(topics_data, dict) and "topics" in topics_data:
            for t in topics_data["topics"]:
                keywords_map[t.get("topic_id", 0)] = t.get("keywords", [])[:5]
        
        for t in result["topic_metrics"]["top_5_topics"]:
            t["keywords"] = keywords_map.get(t["topic_id"], [])
    
    return result


def analyze_experiment(exp_name, threshold):
    """Analizza un intero esperimento."""
    
    base_dir = f"../../results/experiments/{exp_name}"
    
    if not os.path.exists(base_dir):
        print(f"  Esperimento non trovato: {exp_name}")
        return None
    
    print_section(f"ESPERIMENTO: {exp_name} (soglia {threshold*100:.0f}%)")
    
    # Load pipeline config
    config_file = f"{base_dir}/pipeline_config.json"
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
        print(f"\nConfigurazione:")
        print(f"  Soglia: {config.get('threshold', 'N/A')}")
        print(f"  Date: {config.get('start_date', 'N/A')} - {config.get('end_date', 'N/A')}")
        print(f"  Max livelli: {config.get('max_levels', 'N/A')}")
    
    # Load pipeline summary
    summary_file = f"{base_dir}/pipeline_summary.json"
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"\nRiepilogo pipeline:")
        print(f"  Livelli processati: {summary.get('levels_processed', 'N/A')}")
        print(f"  Tempo totale: {summary.get('total_time_seconds', 0)/3600:.2f} ore")
    
    # Find all levels
    level_dirs = glob(f"{base_dir}/level_*")
    levels = []
    for d in level_dirs:
        level_name = os.path.basename(d).replace("level_", "")
        if level_name.isdigit():
            levels.append(int(level_name))
    levels = sorted(levels)
    
    print(f"\nLivelli trovati: {levels}")
    
    # Analyze each level
    all_results = {}
    totals = {
        "channels": 0,
        "political_channels": 0,
        "messages": 0,
        "political_messages": 0
    }
    
    for level in levels:
        print_subsection(f"LIVELLO {level}")
        
        result = analyze_level_detailed(exp_name, level, threshold)
        
        if result is None:
            print("  [SKIP] Dati mancanti")
            continue
        
        all_results[level] = result
        
        # Print preprocessing
        if "preprocessing" in result:
            pp = result["preprocessing"]
            print(f"\n  PREPROCESSING:")
            print(f"    Nodi input: {pp['total_nodes_input']:,}")
            print(f"    Messaggi prima filtro date: {pp['total_messages_before_filter']:,}")
            print(f"    Messaggi dopo filtro date: {pp['total_messages_after_filter']:,}")
            print(f"    Messaggi inglesi finali: {pp['english_messages']:,}")
            print(f"    Canali senza cartella: {pp['channels_no_folder']}")
            print(f"    Canali senza file: {pp['channels_no_files']}")
            print(f"    Canali senza inglese: {pp['channels_no_english']}")
            print(f"    Canali solo msg corti: {pp['channels_only_short']}")
            print(f"    Canali finali: {pp['channels_final']}")
        
        # Print LDA
        if "lda" in result:
            lda = result["lda"]
            print(f"\n  LDA:")
            print(f"    Documenti: {lda['num_documents']:,}")
            print(f"    Vocabolario: {lda['vocab_size']:,}")
            print(f"    Range K: {lda['k_range']}")
            print(f"    K ottimale: {lda['num_topics']}")
            print(f"    Coerenza: {lda['coherence']:.4f}" if lda['coherence'] != float('inf') else f"    Coerenza: inf")
        
        # Print classification
        if "classification" in result:
            cl = result["classification"]
            print(f"\n  CLASSIFICAZIONE TOPIC:")
            print(f"    Topic totali: {cl['total_topics']}")
            print(f"    Topic politici: {cl['political_topics']} ({cl['pct_political_topics']:.1f}%)")
            print(f"    Topic non-politici: {cl['non_political_topics']}")
        
        # Print messages
        if "messages" in result:
            msg = result["messages"]
            print(f"\n  MESSAGGI:")
            print(f"    Totali: {msg['total']:,}")
            print(f"    Politici: {msg['political']:,} ({msg['pct_political']:.1f}%)")
            print(f"    Non-politici: {msg['non_political']:,}")
            totals["messages"] += msg["total"]
            totals["political_messages"] += msg["political"]
        
        # Print channels
        if "channels" in result:
            ch = result["channels"]
            print(f"\n  CANALI:")
            print(f"    Totali: {ch['total']:,}")
            print(f"    Politici: {ch['political']:,} ({ch['pct_political']:.1f}%)")
            print(f"    Non-politici: {ch['non_political']:,}")
            totals["channels"] += ch["total"]
            totals["political_channels"] += ch["political"]
        
        # Print political channels metrics
        if "political_channels_metrics" in result:
            pcm = result["political_channels_metrics"]
            print(f"\n  METRICHE CANALI POLITICI:")
            print(f"    Media msg/canale: {pcm['avg_messages']:.1f}")
            print(f"    Media msg politici/canale: {pcm['avg_political_messages']:.1f}")
            print(f"    Media msg non-pol/canale: {pcm['avg_non_political_messages']:.1f}")
            print(f"    Mediana msg/canale: {pcm['median_messages']:.1f}")
            print(f"    Min-Max msg: {pcm['min_messages']} - {pcm['max_messages']}")
            print(f"    Media ratio politico: {pcm['avg_political_ratio']:.1f}%")
        
        # Print non-political channels metrics
        if "non_political_channels_metrics" in result:
            npcm = result["non_political_channels_metrics"]
            print(f"\n  METRICHE CANALI NON-POLITICI:")
            print(f"    Media msg/canale: {npcm['avg_messages']:.1f}")
            print(f"    Media msg politici/canale: {npcm['avg_political_messages']:.1f}")
            print(f"    Media msg non-pol/canale: {npcm['avg_non_political_messages']:.1f}")
            print(f"    Mediana msg/canale: {npcm['median_messages']:.1f}")
            print(f"    Min-Max msg: {npcm['min_messages']} - {npcm['max_messages']}")
            print(f"    Media ratio politico: {npcm['avg_political_ratio']:.1f}%")
        
        # Print topic metrics
        if "topic_metrics" in result:
            tm = result["topic_metrics"]
            print(f"\n  METRICHE TOPIC:")
            print(f"    Msg in topic politici: {tm['political_topics_total_messages']:,}")
            print(f"    Msg in topic non-pol: {tm['non_political_topics_total_messages']:,}")
            print(f"    Media msg/topic politico: {tm['avg_messages_per_political_topic']:.1f}")
            print(f"    Media msg/topic non-pol: {tm['avg_messages_per_non_political_topic']:.1f}")
            print(f"\n  TOP 5 TOPIC:")
            for t in tm["top_5_topics"]:
                pol_marker = "[P]" if t["is_political"] else "[ ]"
                kw = ", ".join(t.get("keywords", [])[:5])
                print(f"    Topic {t['topic_id']:3} {pol_marker}: {t['messages']:>8,} msg - {kw}")
    
    # Print totals
    print_subsection("TOTALI ESPERIMENTO")
    print(f"  Canali totali: {totals['channels']:,}")
    print(f"  Canali politici: {totals['political_channels']:,} ({100*totals['political_channels']/totals['channels']:.1f}%)" if totals['channels'] > 0 else "")
    print(f"  Messaggi totali: {totals['messages']:,}")
    print(f"  Messaggi politici: {totals['political_messages']:,} ({100*totals['political_messages']/totals['messages']:.1f}%)" if totals['messages'] > 0 else "")
    
    return all_results


def generate_latex_table_data(all_experiments):
    """Genera dati pronti per tabelle LaTeX."""
    
    print_section("DATI PER TABELLE LATEX")
    
    # Tabella confronto soglie
    print_subsection("Tabella confronto soglie")
    print("Threshold | Livelli | Canali | Politici | % Pol | Messaggi | Msg Pol | % Msg Pol")
    print("-" * 90)
    
    for exp_name, data in sorted(all_experiments.items()):
        if data is None:
            continue
        
        threshold = int(exp_name.split("_")[1]) if "threshold_" in exp_name else 40
        num_levels = len(data)
        
        total_ch = sum(d.get("channels", {}).get("total", 0) for d in data.values())
        pol_ch = sum(d.get("channels", {}).get("political", 0) for d in data.values())
        total_msg = sum(d.get("messages", {}).get("total", 0) for d in data.values())
        pol_msg = sum(d.get("messages", {}).get("political", 0) for d in data.values())
        
        pct_ch = 100 * pol_ch / total_ch if total_ch > 0 else 0
        pct_msg = 100 * pol_msg / total_msg if total_msg > 0 else 0
        
        print(f"{threshold:>9}% | {num_levels:>7} | {total_ch:>6,} | {pol_ch:>8,} | {pct_ch:>5.1f}% | {total_msg:>8,} | {pol_msg:>7,} | {pct_msg:>9.1f}%")


def main():
    print("=" * 80)
    print(" ESTRAZIONE DATI PER TESI")
    print(" Esperimenti: threshold_20, threshold_40, threshold_60, threshold_80")
    print("=" * 80)
    
    experiments = [
        ("threshold_20", 0.2),
        ("threshold_40", 0.4),
        ("threshold_60", 0.6),
        ("threshold_80", 0.8),
    ]
    
    all_results = {}
    
    for exp_name, threshold in experiments:
        result = analyze_experiment(exp_name, threshold)
        all_results[exp_name] = result
    
    # Generate summary tables
    generate_latex_table_data(all_results)
    
    # Save complete data to JSON
    output_file = "../../results/thesis_data_complete.json"
    
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj
    
    all_results_clean = convert_types(all_results)
    
    with open(output_file, 'w') as f:
        json.dump(all_results_clean, f, indent=2)
    
    print(f"\n\nDati completi salvati in: {output_file}")


if __name__ == "__main__":
    main()
