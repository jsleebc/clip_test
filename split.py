import os
import random
import shutil

# Set paths
base_path = os.path.expanduser('~/nft_image_test')
source_image_dir = os.path.join(base_path, 'train_image')
source_metadata_dir = os.path.join(base_path, 'train_metadata')
dest_image_dir = os.path.join(base_path, 'test_image')
dest_metadata_dir = os.path.join(base_path, 'test_metadata')

# Create destination directories if they don't exist
os.makedirs(dest_image_dir, exist_ok=True)
os.makedirs(dest_metadata_dir, exist_ok=True)

# Get list of image files and randomly select 1999
image_files = [f for f in os.listdir(source_image_dir) if f.startswith('bayc_')]
selected_files = random.sample(image_files, 1999)

# Move files
for filename in selected_files:
    # Move image file
    image_source = os.path.join(source_image_dir, filename)
    image_dest = os.path.join(dest_image_dir, filename)
    shutil.move(image_source, image_dest)
    
    # Move corresponding metadata file (assuming .json extension)
    metadata_filename = os.path.splitext(filename)[0] + '.json'  # Change extension to .json
    metadata_source = os.path.join(source_metadata_dir, metadata_filename)
    metadata_dest = os.path.join(dest_metadata_dir, metadata_filename)
    shutil.move(metadata_source, metadata_dest)

print(f"Moved {len(selected_files)} files to test directories")