#!/usr/bin/env python3
"""
Quick analysis of clustering levels and overlaps.
"""

import json
from collections import defaultdict

def load_conversations_data(file_path: str) -> dict:
    """Load conversations and create mappings."""
    chat_id_to_question_id = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                conv = json.loads(line)
                chat_id = conv['chat_id']
                question_id = conv['metadata']['question_id']
                chat_id_to_question_id[chat_id] = question_id
    
    return chat_id_to_question_id

def load_clusters_data(file_path: str) -> list:
    """Load cluster data."""
    clusters = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                clusters.append(json.loads(line))
    return clusters

def analyze_levels_and_overlaps(clusters, chat_id_to_question_id):
    """Analyze levels and find overlaps."""
    
    # Group clusters by level
    clusters_by_level = defaultdict(list)
    for cluster in clusters:
        level = cluster.get('level', 0)
        clusters_by_level[level].append(cluster)
    
    print(f"📊 TOTAL LEVELS: {len(clusters_by_level)}")
    print(f"📊 LEVEL BREAKDOWN:")
    for level in sorted(clusters_by_level.keys()):
        print(f"   • Level {level}: {len(clusters_by_level[level])} clusters")
    
    print(f"\n🚨 OVERLAP ANALYSIS:")
    
    total_spillovers = 0
    
    for level in sorted(clusters_by_level.keys()):
        level_clusters = clusters_by_level[level]
        
        # Track which questions appear in which clusters
        question_to_clusters = defaultdict(list)
        
        for cluster in level_clusters:
            cluster_id = cluster['id']
            cluster_name = cluster['name']
            
            for chat_id in cluster['chat_ids']:
                if chat_id in chat_id_to_question_id:
                    question_id = chat_id_to_question_id[chat_id]
                    question_to_clusters[question_id].append({
                        'cluster_id': cluster_id,
                        'cluster_name': cluster_name
                    })
        
        # Find spillover questions
        spillover_questions = {
            q: clusters for q, clusters in question_to_clusters.items() 
            if len(clusters) > 1
        }
        
        if spillover_questions:
            print(f"\n🏷️  LEVEL {level}: {len(spillover_questions)} spillover questions")
            for question_id, cluster_info in spillover_questions.items():
                print(f"   • Question {question_id}: {len(cluster_info)} clusters")
            total_spillovers += len(spillover_questions)
        else:
            print(f"\n🏷️  LEVEL {level}: ✅ No spillovers")
    
    print(f"\n📈 SUMMARY:")
    print(f"   • Total questions with spillovers across all levels: {total_spillovers}")
    
    return clusters_by_level

def main():
    conversations_file = "checkpoints_mt_bench_10k/conversations.jsonl"
    clusters_file = "checkpoints_mt_bench_10k/dimensionality.jsonl"
    
    print("Loading data...")
    chat_id_to_question_id = load_conversations_data(conversations_file)
    clusters = load_clusters_data(clusters_file)
    
    print(f"Loaded {len(chat_id_to_question_id)} conversations and {len(clusters)} clusters\n")
    
    analyze_levels_and_overlaps(clusters, chat_id_to_question_id)

if __name__ == "__main__":
    main()
