import json
import os
import pandas as pd
from collections import defaultdict

def analyze_metadata_directory(directory_path):
    # Initialize data structures
    trait_types = defaultdict(lambda: defaultdict(int))
    total_count = 0
    
    # Process each JSON file
    for filename in os.listdir(directory_path):
        if not filename.endswith('.json'):
            continue
            
        with open(os.path.join(directory_path, filename), 'r') as f:
            data = json.load(f)
            total_count += 1
            
            for trait in data['attributes']:
                trait_types[trait['trait_type']][trait['value']] += 1
    
    # Calculate statistics
    stats = []
    for trait_type, values in trait_types.items():
        for value, count in values.items():
            percentage = (count / total_count) * 100
            stats.append({
                'Trait Type': trait_type,
                'Value': value,
                'Count': count,
                'Percentage': round(percentage, 2)
            })
    
    return stats

def save_statistics(stats, output_prefix):
    # Convert to DataFrame
    df = pd.DataFrame(stats)
    
    # Save as CSV
    df.to_csv(f'{output_prefix}_traits.csv', index=False)
    
    # Save as Excel
    df.to_excel(f'{output_prefix}_traits.xlsx', index=False)
    
    # Save as JSON
    trait_summary = {
        'total_traits': len(set(df['Trait Type'])),
        'traits': {}
    }
    
    for trait_type in df['Trait Type'].unique():
        trait_data = df[df['Trait Type'] == trait_type]
        trait_summary['traits'][trait_type] = {
            'total_values': len(trait_data),
            'values': [
                {
                    'value': row['Value'],
                    'count': row['Count'],
                    'percentage': row['Percentage']
                }
                for _, row in trait_data.iterrows()
            ]
        }
    
    with open(f'{output_prefix}_traits.json', 'w') as f:
        json.dump(trait_summary, f, indent=2)

# Analyze both directories
base_path = os.path.expanduser('~/nft_image_test')

# Process train metadata
train_stats = analyze_metadata_directory(os.path.join(base_path, 'train_metadata'))
save_statistics(train_stats, os.path.join(base_path, 'train'))

# Process test metadata
test_stats = analyze_metadata_directory(os.path.join(base_path, 'test_metadata'))
save_statistics(test_stats, os.path.join(base_path, 'test'))

print("Analysis complete. Files generated for both train and test sets.")