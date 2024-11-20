import json
import os

def generate_clip_description(attributes):
    # Extract trait values
    traits = {attr['trait_type'].lower(): attr['value'] for attr in attributes}
    
    # Start with base description
    desc = "The Bored Ape Yacht Club is a unique digital collectible NFT. "
    
    # Add trait descriptions
    trait_desc = []
    
    if 'background' in traits:
        trait_desc.append(f"has {traits['background'].lower()} background")
    if 'fur' in traits:
        trait_desc.append(f"{traits['fur'].lower()} fur")
    if 'eyes' in traits:
        trait_desc.append(f"{traits['eyes'].lower()} eyes")
    if 'mouth' in traits:
        trait_desc.append(f"{traits['mouth'].lower()} mouth")
    if 'clothes' in traits:
        trait_desc.append(f"wearing {traits['clothes'].lower()}")
    if 'earring' in traits:
        trait_desc.append(f"with {traits['earring'].lower()} earring")
    if 'hat' in traits:
        trait_desc.append(f"wearing {traits['hat'].lower()} hat")
    
    if trait_desc:
        desc += "This ape " + ", ".join(trait_desc) + "."
    
    return desc

def process_train_metadata():
    base_path = os.path.expanduser('~/nft_image_test')
    metadata_path = os.path.join(base_path, 'train_metadata')
    descriptions = {}
    
    for filename in os.listdir(metadata_path):
        if filename.endswith('.json'):
            with open(os.path.join(metadata_path, filename), 'r') as f:
                data = json.load(f)
                nft_id = filename.replace('.json', '')
                descriptions[nft_id] = generate_clip_description(data['attributes'])
    
    output_path = os.path.join(base_path, 'clip_descriptions.json')
    with open(output_path, 'w') as f:
        json.dump(descriptions, f, indent=2)

process_train_metadata()
print("CLIP descriptions generated in clip_descriptions.json")