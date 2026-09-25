import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def generate_mock_oct_scan(output_path, disease_type="healthy"):
    """
    Generates a synthetic OCT cross-section image (512x256).
    Draws horizontal retinal bands with curves and noise.
    Optionally injects visual anomalies corresponding to disease_type.
    Returns: List of annotated bounding boxes in coordinates [x1, y1, x2, y2].
    """
    width, height = 512, 256
    # Create dark background (representing vitreous space)
    img = np.zeros((height, width), dtype=np.uint8) + random.randint(15, 25)
    
    # Layer lines (y coordinates across x)
    x = np.arange(width)
    
    # Base curve using sine waves
    base_y = 120 + 8 * np.sin(x / 40.0) + 2 * np.sin(x / 10.0)
    
    # Define layer offsets
    ilm_y = base_y
    opl_y = base_y + 25
    isos_y = base_y + 40
    rpe_y = base_y + 48
    choroid_y = base_y + 65
    
    bboxes = []
    
    # Apply pathogical deviations to layers
    if disease_type == "PED":
        # Pigment Epithelial Detachment: RPE layer is elevated upward in a dome shape in the middle
        dome_start = 200
        dome_end = 320
        dome_center = 260
        dome_height = 30
        
        # Modify rpe and choroid layers to dome upward
        for idx in range(width):
            if dome_start <= idx <= dome_end:
                factor = np.sin(np.pi * (idx - dome_start) / (dome_end - dome_start))
                rpe_y[idx] -= dome_height * factor
                choroid_y[idx] -= dome_height * factor
                
        # Bounding box for the PED lesion
        # Coordinates: [x1, y1, x2, y2]
        x1, x2 = dome_start, dome_end
        y1 = int(min(rpe_y[dome_start:dome_end])) - 5
        y2 = int(max(rpe_y[dome_start:dome_end])) + 15
        bboxes.append({
            "coords": [x1, y1, x2, y2],
            "label": "PED",
            "description": "Pigment epithelial detachment showing elevated RPE boundary"
        })
        
    elif disease_type == "SRF":
        # Subretinal Fluid: hyporeflective pocket beneath the sensory retina, above RPE
        pocket_start = 180
        pocket_end = 340
        pocket_height = 20
        
        # Retina is pushed up, RPE stays relatively flat
        for idx in range(width):
            if pocket_start <= idx <= pocket_end:
                factor = np.sin(np.pi * (idx - pocket_start) / (pocket_end - pocket_start))
                ilm_y[idx] -= pocket_height * factor
                opl_y[idx] -= pocket_height * factor
                isos_y[idx] -= pocket_height * factor
                
        x1, x2 = pocket_start, pocket_end
        y1 = int(min(isos_y[pocket_start:pocket_end])) - 5
        y2 = int(max(rpe_y[pocket_start:pocket_end])) + 5
        bboxes.append({
            "coords": [x1, y1, x2, y2],
            "label": "SRF",
            "description": "Hyporeflective subretinal fluid accumulation"
        })
        
    elif disease_type == "IRF":
        # Intraretinal Fluid: cystic pockets inside the retina layers (above IS/OS)
        cyst_centers = [(220, 110), (260, 105), (300, 115)]
        for cx, cy in cyst_centers:
            x1, y1, x2, y2 = cx - 12, cy - 8, cx + 12, cy + 8
            bboxes.append({
                "coords": [x1, y1, x2, y2],
                "label": "IRF",
                "description": "Intraretinal fluid cystoid space"
            })
            
    elif disease_type == "MH":
        # Macular Hole: gap/break in the retinal tissue
        hole_start = 240
        hole_end = 272
        
        # Attenuate the visual layers in the gap area
        # We will clear or dim layers here
        x1, x2 = hole_start, hole_end
        y1 = int(min(ilm_y[hole_start:hole_end])) - 10
        y2 = int(max(rpe_y[hole_start:hole_end])) + 5
        bboxes.append({
            "coords": [x1, y1, x2, y2],
            "label": "MH",
            "description": "Full-thickness macular hole showing tissue defect"
        })

    # Render layers into numpy array
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # 1. Vitreous is background
    # 2. Draw Retina body (between ILM and IS/OS) with intermediate brightness
    for idx in range(width - 1):
        # Draw vertical slices for anatomical layers
        draw.line([(idx, int(ilm_y[idx])), (idx, int(rpe_y[idx]))], fill=50)
        draw.line([(idx, int(opl_y[idx])), (idx, int(opl_y[idx]) + 2)], fill=90) # inner layers
        
    # Draw fluid cystoids (IRF) as dark blobs inside the retina if IRF case
    if disease_type == "IRF":
        for box in bboxes:
            bx1, by1, bx2, by2 = box["coords"]
            draw.ellipse([bx1, by1, bx2, by2], fill=20)
            
    # Draw subretinal fluid (SRF) as dark pocket if SRF case
    if disease_type == "SRF":
        # Fill pocket area between isos_y and rpe_y
        for idx in range(pocket_start, pocket_end):
            draw.line([(idx, int(isos_y[idx])), (idx, int(rpe_y[idx]))], fill=15)
            
    # Draw Macular Hole (MH) break
    if disease_type == "MH":
        for idx in range(hole_start, hole_end):
            # Erase retinal tissue down to RPE
            draw.line([(idx, int(ilm_y[idx])), (idx, int(rpe_y[idx] - 2))], fill=25)
            
    # Draw RPE - Retinal Pigment Epithelium (thick highly reflective band)
    for idx in range(width - 1):
        y_val = int(rpe_y[idx])
        draw.line([(idx, y_val), (idx, y_val + 3)], fill=210)
        
    # Draw IS/OS - Photoreceptor Inner/Outer Segment junction (thin reflective band)
    for idx in range(width - 1):
        y_val = int(isos_y[idx])
        draw.line([(idx, y_val), (idx, y_val + 1)], fill=150)
        
    # Draw ILM - Inner Limiting Membrane (thin bright band)
    for idx in range(width - 1):
        y_val = int(ilm_y[idx])
        draw.line([(idx, y_val), (idx, y_val + 1)], fill=180)
        
    # Draw Choroid - Grainy, textured thick structure below RPE
    for idx in range(width - 1):
        rpe_bottom = int(rpe_y[idx]) + 4
        # Draw choroidal columns with random noise texture
        for y_coord in range(rpe_bottom, height):
            # Grainy texture drops off with depth
            depth_factor = max(0, 1.0 - (y_coord - rpe_bottom) / 60.0)
            pixel_val = int((random.randint(60, 120) * depth_factor) + 15)
            draw.point((idx, y_coord), fill=pixel_val)

    # Apply blur filter to make layers smooth and scan-like
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_img.save(output_path)
    
    return bboxes

def main():
    print("Generating mock OCT scans...")
    data_dir = "./data/raw"
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 20 samples with different diagnoses
    diagnoses = ["healthy", "PED", "SRF", "IRF", "MH"]
    dataset_sources = ["OCT5k", "OIMHS", "AMD-SD", "OCTDL"]
    
    samples = []
    
    for i in range(20):
        img_id = f"OCT_SCAN_{i+1:03d}"
        disease = random.choice(diagnoses)
        source = random.choice(dataset_sources)
        img_name = f"{img_id}.png"
        img_path = os.path.join(images_dir, img_name)
        
        # Map visual disease to diagnostic class
        diag_class = "NORMAL"
        if disease == "PED":
            diag_class = "AMD"  # PED is typical in Wet AMD
        elif disease == "SRF":
            diag_class = "CSC"  # Central Serous Chorioretinopathy
        elif disease == "IRF":
            diag_class = "DME"  # Diabetic Macular Edema
        elif disease == "MH":
            diag_class = "MH"   # Macular Hole
            
        # Draw and get boxes
        bboxes = generate_mock_oct_scan(img_path, disease_type=disease)
        
        sample = {
            "image_id": img_id,
            "image_path": img_path,
            "dataset_source": source,
            "width": 512,
            "height": 256,
            "disease_label": diag_class,
            "bboxes": bboxes,
            "clinical_attributes": {
                "age": random.randint(50, 80),
                "gender": random.choice(["Male", "Female"]),
                "eye": random.choice(["OD", "OS"]),
                "scan_quality": random.choice(["High", "Medium"]),
                "visual_acuity": f"20/{random.choice([20, 30, 40, 60, 80])}"
            }
        }
        samples.append(sample)
        print(f"Generated {img_id} ({disease} -> {diag_class}) at {img_path}")
        
    # Save annotations index
    ann_path = os.path.join(data_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved synthetic annotations.json index at {ann_path}")

if __name__ == "__main__":
    main()
