import kagglehub
import os
import shutil

def download_visual_datasets():
    print("Downloading Visual Datasets, Please Wait...")

    base_dir = "./visual_datasets"
    if not os.path.exists(base_dir):
      os.makedirs(base_dir)

    # DATASETS FROM KAGGLE
    datasets = {
        "1_COLD": "puneet6060/intel-image-classification",
        "2_HEAT": "aseem001/natural-disaster-images-dataset",
        "3_TOXIN": "asdasdasasdas/garbage-classification",
        "4_SCARCITY": "cdminix/us-drought-meteorological-data", 
        "5_AIRLESS": "greatsharma/underwater-image-classification" 
    }

    for category, dataset_handle in datasets.items():

      target_path = os.path.join(base_dir, category)

      if os.path.exists(target_path):
            print(f"{category} was Loaded Already, Download is Stopped.")
            continue

      print(f"{category} is Downloading...")

      try:
       path = kagglehub.dataset_download(dataset_handle)

       print(f"The Files are Being Moved into the: {target_path}...")
       shutil.copytree(path, target_path, dirs_exist_ok=True)
      except Exception as e:
       print(f"{category} can't downloading. Problem: {e}")

    print("\nSetup is Finished.")

if __name__ == "__main__":
  download_visual_datasets()

