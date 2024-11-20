import os
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import random
from tqdm import tqdm
import datetime

class NFTDataset(Dataset):
    def __init__(self, image_dir, metadata_dir, image_files, processor):
        self.image_dir = image_dir
        self.metadata_dir = metadata_dir
        self.image_files = image_files
        self.processor = processor
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        metadata_path = os.path.join(self.metadata_dir, image_name.replace('.png', '.json'))
        
        image = Image.open(image_path).convert('RGB')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            text = self.create_description(metadata)
        
        return image, text
    
    @staticmethod
    def create_description(metadata):
        try:
            attributes = metadata.get('attributes', [])
            description = "The Bored Ape Yacht Club is a unique digital collectible NFT. This ape"
            
            attr_dict = {attr['trait_type']: attr['value'] for attr in attributes}
            
            if 'Background' in attr_dict:
                description += f" has {attr_dict['Background'].lower()} background"
            if 'Fur' in attr_dict:
                description += f", {attr_dict['Fur'].lower()} fur"
            if 'Eyes' in attr_dict:
                description += f", {attr_dict['Eyes'].lower()} eyes"
            if 'Mouth' in attr_dict:
                description += f", {attr_dict['Mouth'].lower()} mouth"
            if 'Clothes' in attr_dict:
                description += f", wearing {attr_dict['Clothes'].lower()}"
            if 'Earring' in attr_dict:
                description += f", with {attr_dict['Earring'].lower()} earring"
            if 'Hat' in attr_dict:
                description += f", wearing {attr_dict['Hat'].lower()} hat"
            
            description += "."
            return description
        except Exception as e:
            print(f"Error creating description: {e}")
            return "A Bored Ape NFT"

def collate_fn(batch):
    images, texts = zip(*batch)
    inputs = processor(
        images=list(images),
        text=list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    return inputs

def compute_loss(outputs, device):
    batch_size = outputs.logits_per_image.shape[0]
    labels = torch.arange(batch_size).to(device)
    
    loss_i2t = F.cross_entropy(outputs.logits_per_image, labels)
    loss_t2i = F.cross_entropy(outputs.logits_per_text, labels)
    
    return (loss_i2t + loss_t2i) / 2

def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    
    for idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = compute_loss(outputs, device)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        current_avg_loss = total_loss / (idx + 1)
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{current_avg_loss:.4f}",
            'batch': f"{idx + 1}/{len(train_loader)}"
        })
    
    return total_loss / len(train_loader)

def main():
    print("Starting script...")
    
    # Updated paths
    base_dir = os.path.expanduser("~/nft_image_test")
    train_image_dir = os.path.join(base_dir, "train_image")
    train_metadata_dir = os.path.join(base_dir, "train_metadata")
    save_dir = os.path.join(base_dir, "model_results")
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("\nLoading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    global processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    train_files = [f for f in os.listdir(train_image_dir) if f.startswith('bayc_') and f.endswith('.png')]
    print(f"\nTraining images: {len(train_files)}")
    
    train_dataset = NFTDataset(train_image_dir, train_metadata_dir, train_files, processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5
    
    print("\nStarting training...")
    try:
        best_loss = float('inf')
        training_history = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch + 1)
            
            training_history.append({
                'epoch': epoch + 1,
                'avg_loss': float(avg_loss)
            })
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(save_dir, f"best_model_{timestamp}")
                model.save_pretrained(best_model_path)
                processor.save_pretrained(best_model_path)
                print(f"Saved best model with loss {best_loss:.4f}")
            
            epoch_save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}_{timestamp}")
            model.save_pretrained(epoch_save_path)
            processor.save_pretrained(epoch_save_path)
        
        history_file = os.path.join(save_dir, f'training_history_{timestamp}.json')
        with open(history_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'training_history': training_history,
                'best_loss': float(best_loss)
            }, f, indent=2)

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()