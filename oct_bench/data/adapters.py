import os
import json
from typing import List
from .schemas import ImageSample, AnnotationBox, AnnotationMask

class BaseAdapter:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir

    def load_samples(self) -> List[ImageSample]:
        raise NotImplementedError("Each adapter must implement load_samples method.")

class UnifiedJSONAdapter(BaseAdapter):
    """
    Adapter that parses a unified annotations.json index (e.g. generated mock data)
    """
    def __init__(self, raw_data_dir: str, json_name: str = "annotations.json"):
        super().__init__(raw_data_dir)
        self.json_path = os.path.join(raw_data_dir, json_name)

    def load_samples(self) -> List[ImageSample]:
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Annotations index not found at {self.json_path}")
            
        with open(self.json_path, "r") as f:
            data = json.load(f)
            
        samples = []
        for item in data:
            # Reconstruct absolute/relative paths based on raw_data_dir
            orig_path = item["image_path"]
            # Convert standard raw paths if they are windows/unix mismatched
            clean_path = orig_path.replace("\\", "/")
            if not os.path.isabs(clean_path):
                # Ensure path is relative to target raw data dir
                # If path contains './data/raw' or similar, normalize it
                if clean_path.startswith("./data/raw/"):
                    clean_path = clean_path.replace("./data/raw/", "")
                clean_path = os.path.join(self.raw_data_dir, clean_path)

            bboxes = []
            for box in item.get("bboxes", []):
                bboxes.append(AnnotationBox(
                    coords=box["coords"],
                    label=box["label"],
                    description=box.get("description")
                ))

            masks = []
            for mask in item.get("masks", []):
                masks.append(AnnotationMask(
                    mask_path=mask["mask_path"],
                    label=mask["label"],
                    color=mask.get("color")
                ))

            sample = ImageSample(
                image_id=item["image_id"],
                image_path=clean_path,
                dataset_source=item["dataset_source"],
                width=item["width"],
                height=item["height"],
                disease_label=item.get("disease_label"),
                bboxes=bboxes,
                masks=masks,
                clinical_attributes=item.get("clinical_attributes", {})
            )
            samples.append(sample)
            
        return samples

# Future placeholder adapters for direct datasets:
class OCT5kAdapter(BaseAdapter):
    def load_samples(self) -> List[ImageSample]:
        # Implementation for raw OCT5k directory structures (e.g. folder names are labels)
        samples = []
        # Custom logic goes here
        return samples

class OIMHSAdapter(BaseAdapter):
    def load_samples(self) -> List[ImageSample]:
        # Implementation for OIMHS directory (matlab files, label files)
        samples = []
        # Custom logic goes here
        return samples


class HuggingFaceDatasetAdapter(BaseAdapter):
    """
    Adapter to download and parse image datasets directly from Hugging Face Hub.
    Downloads dataset rows and dumps PIL images locally to register them in the pipeline.
    """
    def __init__(
        self, 
        raw_data_dir: str, 
        dataset_name: str, 
        split: str = "train", 
        image_column: str = "image", 
        label_column: str = "label"
    ):
        super().__init__(raw_data_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.image_column = image_column
        self.label_column = label_column
        self.local_images_dir = os.path.join(raw_data_dir, "hf_images")
        os.makedirs(self.local_images_dir, exist_ok=True)

    def load_samples(self) -> List[ImageSample]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets (pip install datasets) to use the HuggingFaceDatasetAdapter.")
            
        print(f"Connecting to Hugging Face Hub to load dataset '{self.dataset_name}' (split: '{self.split}')...")
        dataset = load_dataset(self.dataset_name, split=self.split)
        
        samples = []
        for idx, row in enumerate(dataset):
            img_id = f"HF_SCAN_{idx+1:04d}"
            
            # Save PIL image locally to raw data folder
            pil_img = row[self.image_column]
            local_img_path = os.path.join(self.local_images_dir, f"{img_id}.png")
            pil_img.save(local_img_path)
            
            # Translate disease label classification
            raw_label = row.get(self.label_column, "NORMAL")
            # Map integer category index to name string if metadata is present
            if hasattr(dataset.features.get(self.label_column), "names"):
                disease_label = dataset.features[self.label_column].names[raw_label]
            else:
                disease_label = str(raw_label)
                
            sample = ImageSample(
                image_id=img_id,
                image_path=local_img_path.replace("\\", "/"),
                dataset_source=f"HF_{self.dataset_name.split('/')[-1]}",
                width=pil_img.width,
                height=pil_img.height,
                disease_label=disease_label,
                bboxes=[],
                masks=[],
                clinical_attributes={"hf_row_index": idx}
            )
            samples.append(sample)
            
        return samples

