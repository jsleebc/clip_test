import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np

def get_latest_model_path(base_path):
    model_dir = os.path.join(base_path, "model_results")
    model_folders = [f for f in os.listdir(model_dir) if f.startswith('best_model_')]
    if not model_folders:
        raise ValueError("No model found in model_results directory")
    latest_model = sorted(model_folders)[-1]
    return os.path.join(model_dir, latest_model)

def get_image_embeddings(model, processor, image_path, device):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().detach().numpy()

def compute_text_similarity(model, processor, image_path, text_prompts, device):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(
        images=image,
        text=text_prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply softmax to normalize scores between 0 and 1
        similarities = torch.nn.functional.softmax(outputs.logits_per_image[0], dim=0)
    return similarities.cpu().detach().numpy()

def test_similarities():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = os.path.expanduser("~/nft_image_test")
    model_path = get_latest_model_path(base_path)
    test_image_dir = os.path.join(base_path, "test_image")
    
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.eval()

    general_prompts = [
        "this is nft artwork",
        "this is bored ape yacht club",
        "this is bayc",
        "this is monkey",
        "this is digital art",
        "this is cartoon character"
    ]
    
    trait_prompts = [
        "a bored ape with gold fur",
        "a bored ape with robot fur",
        "a bored ape wearing clothes",
        "a bored ape with laser eyes",
        "a bored ape with rare traits",
        "a bored ape with hat"
    ]

    test_images = [f for f in os.listdir(test_image_dir) if f.startswith('bayc_')]
    
    # Compute all embeddings first
    print("Computing image embeddings...")
    embeddings = {}
    for image_file in tqdm(test_images):
        image_path = os.path.join(test_image_dir, image_file)
        embeddings[image_file] = get_image_embeddings(model, processor, image_path, device)[0]
    
    results = {}
    print("Computing similarities...")
    for image_file in tqdm(test_images):
        current_embedding = embeddings[image_file]
        
        # Compute image similarities
        similarities = []
        for other_file, other_embedding in embeddings.items():
            if other_file != image_file:
                sim = np.dot(current_embedding, other_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(other_embedding)
                )
                similarities.append({'image': other_file, 'similarity': float(sim)})
        
        sorted_similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
        
        # Compute text similarities
        image_path = os.path.join(test_image_dir, image_file)
        general_similarities = compute_text_similarity(model, processor, image_path, general_prompts, device)
        trait_similarities = compute_text_similarity(model, processor, image_path, trait_prompts, device)
        
        results[image_file] = {
            "image_similarity": {
                "top_10": sorted_similarities[:10],
                "bottom_10": sorted_similarities[-10:]
            },
            "text_similarity": {
                "general_recognition": {
                    prompt: float(sim) for prompt, sim in zip(general_prompts, general_similarities)
                },
                "trait_recognition": {
                    prompt: float(sim) for prompt, sim in zip(trait_prompts, trait_similarities)
                }
            }
        }
        
        # Periodically save results
        if len(results) % 100 == 0:
            output_path = os.path.join(base_path, "similarity_test_results.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Final save
    output_path = os.path.join(base_path, "similarity_test_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to similarity_test_results.json")

if __name__ == "__main__":
    torch.cuda.empty_cache()  # Clear CUDA cache before starting
    test_similarities()