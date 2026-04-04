import json
import os
from collections import Counter
import statistics

# Dataset directory
dataset_dir = "SemEval8_Dataset"

# Files to analyze
files = {
    "train": os.path.join(dataset_dir, "subtaskA_train_monolingual.jsonl"),
    "dev": os.path.join(dataset_dir, "subtaskA_dev_monolingual.jsonl")
}

def analyze_dataset(filepath):
    """Analyze a JSONL dataset file and return statistics."""
    records = []
    labels = []
    text_lengths = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records.append(record)
                
                # Extract label (handle different possible label field names)
                if 'label' in record:
                    labels.append(record['label'])
                elif 'labels' in record:
                    labels.append(record['labels'])
                
                # Extract text length (handle different possible text field names)
                text = ""
                if 'text' in record:
                    text = record['text']
                elif 'sentence' in record:
                    text = record['sentence']
                
                if text:
                    text_lengths.append(len(text.split()))  # Word count
    
    return records, labels, text_lengths

# Analyze both files
print("=" * 70)
print("SemEval 8 Dataset Analysis")
print("=" * 70)

total_records = 0
all_labels = []
all_text_lengths = []

for dataset_name, filepath in files.items():
    if os.path.exists(filepath):
        print(f"\n{dataset_name.upper()} DATASET")
        print("-" * 70)
        
        records, labels, text_lengths = analyze_dataset(filepath)
        
        total_records += len(records)
        all_labels.extend(labels)
        all_text_lengths.extend(text_lengths)
        
        # Print statistics
        print(f"Number of records: {len(records)}")
        
        if labels:
            label_counts = Counter(labels)
            print(f"\nLabel distribution:")
            for label, count in sorted(label_counts.items()):
                percentage = (count / len(labels)) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")
        
        if text_lengths:
            print(f"\nText statistics (word count):")
            print(f"  Min words: {min(text_lengths)}")
            print(f"  Max words: {max(text_lengths)}")
            print(f"  Average words: {statistics.mean(text_lengths):.1f}")
            print(f"  Median words: {statistics.median(text_lengths):.1f}")
        
        # Sample record
        if records:
            print(f"\nSample record:")
            sample = records[0]
            for key, value in list(sample.items())[:3]:
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
    else:
        print(f"File not found: {filepath}")

# Overall statistics
print("\n" + "=" * 70)
print("OVERALL STATISTICS")
print("=" * 70)
print(f"Total records (train + dev): {total_records}")

if all_labels:
    label_counts = Counter(all_labels)
    print(f"\nOverall label distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(all_labels)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

if all_text_lengths:
    print(f"\nOverall text statistics (word count):")
    print(f"  Min words: {min(all_text_lengths)}")
    print(f"  Max words: {max(all_text_lengths)}")
    print(f"  Average words: {statistics.mean(all_text_lengths):.1f}")
    print(f"  Median words: {statistics.median(all_text_lengths):.1f}")

print("\n" + "=" * 70)
