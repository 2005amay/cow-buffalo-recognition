import kagglehub
import shutil
import os

def setup_dataset():
    print("Downloading dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("devang03mgr/cows-breed")
    print("Path to dataset files:", path)

    target_dir = r"d:\project k\dataset"
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    print(f"Moving files to {target_dir}...")
    
    # Walk through the downloaded directory and move breed folders
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            source_folder = os.path.join(root, dir_name)
            destination_folder = os.path.join(target_dir, dir_name)
            
            # If the folder is a breed folder (not some internal kaggle folder), move it
            # We assume the dataset structure is straightforward
            try:
                if os.path.exists(destination_folder):
                    print(f"Merging {dir_name}...")
                    # Merge logic if needed, or just copy files
                    for file_name in os.listdir(source_folder):
                        shutil.copy2(os.path.join(source_folder, file_name), destination_folder)
                else:
                    print(f"Moving {dir_name}...")
                    shutil.copytree(source_folder, destination_folder)
            except Exception as e:
                print(f"Error moving {dir_name}: {e}")

    print("Dataset setup complete!")
    print(f"Check your images in: {target_dir}")

if __name__ == "__main__":
    setup_dataset()
