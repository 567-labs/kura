#!/usr/bin/env python3
"""
Analyze cluster distribution from dimensionality checkpoint file.
Evaluates top-level cluster distribution using conversation ID as unique key.
"""

import json
from collections import defaultdict, Counter
from typing import Dict, List, Set
import pandas as pd

def load_dimensionality_data(file_path: str) -> List[Dict]:
    """Load cluster data from JSONL file."""
    clusters = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                clusters.append(json.loads(line))
    return clusters

def load_conversations_data(file_path: str) -> Dict[str, int]:
    """Load conversations and map chat_id to question_id."""
    chat_id_to_question_id = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                conv = json.loads(line)
                chat_id = conv['chat_id']
                question_id = conv['metadata']['question_id']
                chat_id_to_question_id[chat_id] = question_id
    return chat_id_to_question_id

def analyze_question_distribution_by_level(clusters: List[Dict], chat_id_to_question_id: Dict[str, int]) -> Dict:
    """Analyze how questions are distributed across clusters at each level."""
    
    # Group clusters by level
    clusters_by_level = defaultdict(list)
    for cluster in clusters:
        level = cluster.get('level', 0)
        clusters_by_level[level].append(cluster)
    
    level_analysis = {}
    
    for level, level_clusters in clusters_by_level.items():
        print(f"\nAnalyzing level {level} with {len(level_clusters)} clusters...")
        
        # Track which questions appear in which clusters at this level
        question_to_clusters = defaultdict(set)
        cluster_to_questions = defaultdict(set)
        
        for cluster in level_clusters:
            cluster_id = cluster['id']
            cluster_name = cluster['name']
            
            # Get all questions in this cluster
            questions_in_cluster = set()
            for chat_id in cluster['chat_ids']:
                if chat_id in chat_id_to_question_id:
                    question_id = chat_id_to_question_id[chat_id]
                    questions_in_cluster.add(question_id)
                    question_to_clusters[question_id].add((cluster_id, cluster_name))
            
            cluster_to_questions[cluster_id] = {
                'name': cluster_name,
                'questions': questions_in_cluster,
                'count': len(questions_in_cluster)
            }
        
        # Analyze distribution patterns
        questions_in_multiple_clusters = {
            q: clusters for q, clusters in question_to_clusters.items() 
            if len(clusters) > 1
        }
        
        level_analysis[level] = {
            'total_clusters': len(level_clusters),
            'question_to_clusters': dict(question_to_clusters),
            'cluster_to_questions': dict(cluster_to_questions),
            'questions_in_multiple_clusters': questions_in_multiple_clusters,
            'total_unique_questions': len(question_to_clusters),
            'avg_questions_per_cluster': sum(len(q) for q in cluster_to_questions.values()) / len(level_clusters) if level_clusters else 0
        }
    
    return level_analysis

def analyze_cluster_distribution(clusters: List[Dict]) -> Dict:
    """Analyze distribution of conversations across top-level clusters."""
    
    # Filter for top-level clusters (level=0 or parent_id=null)
    top_level_clusters = [c for c in clusters if c.get('level') == 0 or c.get('parent_id') is None]
    
    print(f"Found {len(top_level_clusters)} top-level clusters")
    print(f"Total clusters in file: {len(clusters)}")
    
    # Collect all unique conversation IDs across top-level clusters
    all_chat_ids = set()
    cluster_distributions = {}
    
    for cluster in top_level_clusters:
        cluster_id = cluster['id']
        cluster_name = cluster['name']
        chat_ids = set(cluster['chat_ids'])
        
        cluster_distributions[cluster_id] = {
            'name': cluster_name,
            'description': cluster['description'],
            'slug': cluster['slug'],
            'chat_ids': chat_ids,
            'count': len(chat_ids),
            'level': cluster.get('level', 'unknown'),
            'coordinates': (cluster.get('x_coord'), cluster.get('y_coord'))
        }
        
        all_chat_ids.update(chat_ids)
    
    # Check for overlaps between clusters
    overlaps = defaultdict(list)
    cluster_ids = list(cluster_distributions.keys())
    
    for i, cluster_id_1 in enumerate(cluster_ids):
        for cluster_id_2 in cluster_ids[i+1:]:
            chat_ids_1 = cluster_distributions[cluster_id_1]['chat_ids']
            chat_ids_2 = cluster_distributions[cluster_id_2]['chat_ids']
            
            overlap = chat_ids_1.intersection(chat_ids_2)
            if overlap:
                overlaps[cluster_id_1].append((cluster_id_2, len(overlap)))
                overlaps[cluster_id_2].append((cluster_id_1, len(overlap)))
    
    # Analyze conversation ID patterns
    chat_id_patterns = analyze_chat_id_patterns(all_chat_ids)
    
    return {
        'total_unique_conversations': len(all_chat_ids),
        'total_top_level_clusters': len(top_level_clusters),
        'cluster_distributions': cluster_distributions,
        'overlaps': dict(overlaps),
        'chat_id_patterns': chat_id_patterns
    }

def analyze_chat_id_patterns(chat_ids: Set[str]) -> Dict:
    """Analyze patterns in chat IDs to understand conversation structure."""
    
    # Parse chat_ids to extract question_id, model, and idx
    # Format: {question_id}_{model}_{idx}
    question_ids = []
    models = []
    indices = []
    
    for chat_id in chat_ids:
        parts = chat_id.split('_')
        if len(parts) >= 3:
            question_id = parts[0]
            model = '_'.join(parts[1:-1])  # Handle models with underscores
            idx = parts[-1]
            
            question_ids.append(question_id)
            models.append(model)
            indices.append(idx)
    
    return {
        'unique_question_ids': len(set(question_ids)),
        'unique_models': len(set(models)),
        'model_distribution': Counter(models),
        'question_id_distribution': Counter(question_ids),
        'total_chat_ids': len(chat_ids)
    }

def print_question_distribution_report(level_analysis: Dict):
    """Print question distribution analysis across levels."""
    
    print("\n" + "=" * 100)
    print("QUESTION DISTRIBUTION ACROSS CLUSTER LEVELS")
    print("=" * 100)
    
    for level in sorted(level_analysis.keys()):
        analysis = level_analysis[level]
        print(f"\n🏷️  LEVEL {level}:")
        print(f"   • Total clusters: {analysis['total_clusters']}")
        print(f"   • Unique questions covered: {analysis['total_unique_questions']}")
        print(f"   • Average questions per cluster: {analysis['avg_questions_per_cluster']:.1f}")
        
        # Show questions in multiple clusters
        multi_cluster_questions = analysis['questions_in_multiple_clusters']
        if multi_cluster_questions:
            print(f"   • Questions appearing in multiple clusters: {len(multi_cluster_questions)}")
            print(f"     WARNING: Questions split across multiple clusters at this level!")
            for question_id, cluster_list in list(multi_cluster_questions.items())[:5]:  # Show first 5
                cluster_names = [name for _, name in cluster_list]
                print(f"     - Question {question_id}: {len(cluster_list)} clusters ({', '.join(cluster_names[:2])}{', ...' if len(cluster_names) > 2 else ''})")
        else:
            print(f"   ✅ All questions belong to exactly one cluster")
        
        # Show cluster details for this level
        cluster_info = analysis['cluster_to_questions']
        sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['count'], reverse=True)
        
        print(f"\n   📊 Cluster breakdown:")
        for cluster_id, info in sorted_clusters[:10]:  # Show top 10 clusters
            print(f"     • {info['name']}: {info['count']} questions")
        
        if len(sorted_clusters) > 10:
            print(f"     ... and {len(sorted_clusters) - 10} more clusters")

def print_analysis_report(analysis: Dict):
    """Print a comprehensive analysis report."""
    
    print("=" * 80)
    print("CLUSTER DISTRIBUTION ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\n📊 OVERVIEW:")
    print(f"   • Total unique conversations: {analysis['total_unique_conversations']:,}")
    print(f"   • Total top-level clusters: {analysis['total_top_level_clusters']}")
    
    # Chat ID patterns
    patterns = analysis['chat_id_patterns']
    print(f"\n🔍 CONVERSATION PATTERNS:")
    print(f"   • Unique question IDs: {patterns['unique_question_ids']:,}")
    print(f"   • Unique models: {patterns['unique_models']}")
    print(f"   • Model distribution:")
    for model, count in patterns['model_distribution'].most_common():
        print(f"     - {model}: {count:,}")
    
    # Cluster distribution
    distributions = analysis['cluster_distributions']
    sorted_clusters = sorted(distributions.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print(f"\n📈 TOP-LEVEL CLUSTER DISTRIBUTION:")
    print(f"{'Rank':<4} {'Count':<8} {'%':<6} {'Cluster Name'}")
    print("-" * 80)
    
    total_conversations = analysis['total_unique_conversations']
    for i, (cluster_id, data) in enumerate(sorted_clusters, 1):
        percentage = (data['count'] / total_conversations) * 100
        print(f"{i:<4} {data['count']:<8,} {percentage:<6.1f}% {data['name']}")
    
    # Show overlaps if any
    overlaps = analysis['overlaps']
    if overlaps:
        print(f"\n⚠️  CLUSTER OVERLAPS DETECTED:")
        for cluster_id, overlap_list in overlaps.items():
            cluster_name = distributions[cluster_id]['name']
            print(f"   • {cluster_name} overlaps with:")
            for other_id, overlap_count in overlap_list:
                other_name = distributions[other_id]['name']
                print(f"     - {other_name}: {overlap_count} conversations")
    else:
        print(f"\n✅ NO OVERLAPS: All clusters have distinct conversation sets")
    
    print(f"\n📝 DETAILED CLUSTER INFORMATION:")
    for i, (cluster_id, data) in enumerate(sorted_clusters[:10], 1):  # Top 10
        print(f"\n{i}. {data['name']} (ID: {cluster_id})")
        print(f"   Count: {data['count']:,} conversations")
        print(f"   Description: {data['description'][:200]}...")
        print(f"   Coordinates: {data['coordinates']}")

def main():
    clusters_file = "checkpoints_mt_bench_10k/dimensionality.jsonl"
    conversations_file = "checkpoints_mt_bench_10k/conversations.jsonl"
    
    try:
        print("Loading cluster data...")
        clusters = load_dimensionality_data(clusters_file)
        
        print("Loading conversations data...")
        chat_id_to_question_id = load_conversations_data(conversations_file)
        print(f"Loaded {len(chat_id_to_question_id)} conversation mappings")
        
        print("Analyzing conversation distribution...")
        analysis = analyze_cluster_distribution(clusters)
        
        print("Analyzing question distribution by level...")
        level_analysis = analyze_question_distribution_by_level(clusters, chat_id_to_question_id)
        
        print_analysis_report(analysis)
        print_question_distribution_report(level_analysis)
        
        # Save detailed results to CSV for further analysis
        distributions = analysis['cluster_distributions']
        df_data = []
        for cluster_id, data in distributions.items():
            df_data.append({
                'cluster_id': cluster_id,
                'name': data['name'],
                'description': data['description'],
                'slug': data['slug'],
                'count': data['count'],
                'percentage': (data['count'] / analysis['total_unique_conversations']) * 100,
                'level': data['level'],
                'x_coord': data['coordinates'][0],
                'y_coord': data['coordinates'][1]
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('count', ascending=False)
        df.to_csv('cluster_distribution_analysis.csv', index=False)
        
        # Save question distribution analysis
        question_df_data = []
        for level, level_data in level_analysis.items():
            for cluster_id, cluster_info in level_data['cluster_to_questions'].items():
                question_df_data.append({
                    'level': level,
                    'cluster_id': cluster_id,
                    'cluster_name': cluster_info['name'],
                    'question_count': cluster_info['count'],
                    'questions': ','.join(map(str, sorted(cluster_info['questions'])))
                })
        
        question_df = pd.DataFrame(question_df_data)
        question_df.to_csv('question_distribution_by_level.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   • Conversation analysis: 'cluster_distribution_analysis.csv'")
        print(f"   • Question analysis: 'question_distribution_by_level.csv'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please ensure both checkpoint files exist.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
