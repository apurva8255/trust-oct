import os
import sys

# Ensure current directory is on python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from oct_bench.data import HuggingFaceDatasetAdapter

def main():
    print("Initializing HuggingFaceDatasetAdapter...")
    
    # We load a small validation subset from keremberke's OCT dataset
    adapter = HuggingFaceDatasetAdapter(
        raw_data_dir="./data/raw",
        dataset_name="keremberke/oct-image-classification",
        split="validation",
        image_column="image",
        label_column="label"
    )
    
    try:
        # Load and download the dataset
        samples = adapter.load_samples()
        
        if not samples:
            print("No samples found.")
            return

        print("\n=== Download Success ===")
        print(f"Total samples loaded: {len(samples)}")
        
        # Details of the first sample
        sample = samples[0]
        print(f"Image ID: {sample.image_id}")
        print(f"Path on disk: {sample.image_path}")
        print(f"Disease Class: {sample.disease_label}")
        print(f"Dimensions: {sample.width}x{sample.height}")
        print("=========================")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have installed 'datasets' package (pip install datasets).")

if __name__ == "__main__":
    main()
