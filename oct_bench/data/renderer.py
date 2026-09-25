import os
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class OCTRenderer:
    """
    Utility class to load OCT images and overlay bounding boxes, numbered annotations, 
    or color highlights for anatomical regions.
    """
    def __init__(self, bbox_colors: Optional[Dict[str, str]] = None, default_line_width: int = 3):
        # Default color map if not provided
        self.bbox_colors = bbox_colors or {
            "PED": "red",
            "SRF": "blue",
            "IRF": "purple",
            "SHRM": "orange",
            "MH": "green",
            "default": "yellow"
        }
        self.line_width = default_line_width

    def _load_image(self, image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        return Image.open(image_path).convert("RGB")

    def draw_bboxes(self, image_path: str, bboxes: List[Dict], output_path: str) -> None:
        """
        Draws bounding boxes on the image, using label-specific colors.
        """
        img = self._load_image(image_path)
        draw = ImageDraw.Draw(img)
        
        for box in bboxes:
            coords = box["coords"] # [x1, y1, x2, y2]
            label = box["label"]
            color = self.bbox_colors.get(label, self.bbox_colors.get("default", "yellow"))
            
            draw.rectangle(coords, outline=color, width=self.line_width)
            
        img.save(output_path)

    def draw_numbered_bboxes(self, image_path: str, bboxes: List[Dict], output_path: str) -> None:
        """
        Draws bounding boxes and labels them with sequential numbers (1, 2, 3...) 
        for spatial/scale comparison tasks.
        """
        img = self._load_image(image_path)
        draw = ImageDraw.Draw(img)
        
        for idx, box in enumerate(bboxes):
            coords = box["coords"]
            color = "red" # standard high-visibility color
            draw.rectangle(coords, outline=color, width=self.line_width)
            
            # Draw annotation index text (e.g. "Box 1")
            text = f"Box {idx + 1}"
            x1, y1, x2, y2 = coords
            
            # Draw text background box
            text_size = (50, 15) # approximate size
            draw.rectangle([x1, max(0, y1 - 18), x1 + text_size[0], y1], fill="red")
            draw.text((x1 + 3, max(0, y1 - 16)), text, fill="white")
            
        img.save(output_path)

    def highlight_anatomical_region(
        self, 
        image_path: str, 
        region_name: str, 
        output_path: str, 
        color_rgb: Tuple[int, int, int] = (255, 0, 0),
        alpha: int = 100
    ) -> None:
        """
        Highlights macro anatomical regions (Vitreous, Retina, Choroid) by blending
        a semi-transparent color mask over simulated coordinates.
        """
        img = self._load_image(image_path)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        width, height = img.size
        
        # We can approximate regional structures using our mock generation conventions:
        # Retina starts around y=120 and ends around y=170
        # Choroid is below y=170
        # Vitreous is above y=120
        # For real images, this would use a segmentation mask path.
        
        if region_name.lower() == "vitreous":
            # Highlight top area
            draw.rectangle([0, 0, width, 120], fill=color_rgb + (alpha,))
        elif region_name.lower() == "retina":
            # Highlight middle area
            draw.rectangle([0, 120, width, 168], fill=color_rgb + (alpha,))
        elif region_name.lower() == "choroid":
            # Highlight bottom area
            draw.rectangle([0, 168, width, height], fill=color_rgb + (alpha,))
        else:
            # Highlight entire image if unknown
            draw.rectangle([0, 0, width, height], fill=color_rgb + (alpha,))
            
        # Composite overlay with base image
        img_rgba = img.convert("RGBA")
        result = Image.alpha_composite(img_rgba, overlay).convert("RGB")
        result.save(output_path)

    def highlight_retinal_layer(
        self,
        image_path: str,
        layer_name: str,
        output_path: str,
        color_rgb: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 3
    ) -> None:
        """
        Highlights specific lines (retinal layers like ILM, OPL, IS/OS, RPE) in the image.
        For simulated images, we draw color along the corresponding Y heights.
        """
        img = self._load_image(image_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Layer heights
        y_mapping = {
            "ilm": 120,
            "opl": 145,
            "isos": 160,
            "rpe": 168
        }
        
        target_y = y_mapping.get(layer_name.lower(), 120)
        
        # Draw a curved highlighted line matching our sine wave generator:
        # base_y = 120 + 8 * sin(x/40) + 2 * sin(x/10)
        # We adjust offset based on layer
        offset = target_y - 120
        
        points = []
        for x in range(width):
            y_val = 120 + 8 * np.sin(x / 40.0) + 2 * np.sin(x / 10.0) + offset
            points.append((x, int(y_val)))
            
        draw.line(points, fill=color_rgb, width=thickness)
        img.save(output_path)
