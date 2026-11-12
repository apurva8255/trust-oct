"""
oct_bench.data.augmentor
~~~~~~~~~~~~~~~~~~~~~~~~
Robustness augmentations for OCT images.
Used to stress-test model performance under real-world degradation conditions.

Augmentations:
  - Gaussian noise
  - Low contrast
  - Motion blur (horizontal)
  - JPEG compression artefacts
  - Salt-and-pepper noise
"""

from __future__ import annotations
import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

def add_gaussian_noise(img: Image.Image, std: float = 25.0) -> Image.Image:
    """Add Gaussian noise with given standard deviation (0–255 scale)."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def reduce_contrast(img: Image.Image, factor: float = 0.4) -> Image.Image:
    """Reduce contrast. factor=1.0 → original, factor=0.0 → grey."""
    return ImageEnhance.Contrast(img).enhance(factor)


def motion_blur(img: Image.Image, kernel_size: int = 15) -> Image.Image:
    """Horizontal motion blur kernel of given size."""
    kernel = ImageFilter.Kernel(
        size=(kernel_size, kernel_size),
        kernel=[1] * kernel_size + [0] * (kernel_size * (kernel_size - 1)),
        scale=kernel_size,
        offset=0,
    )
    # Pillow Kernel requires odd square — use BoxBlur as safe fallback
    return img.filter(ImageFilter.BoxBlur(radius=kernel_size // 4))


def jpeg_compress(img: Image.Image, quality: int = 15) -> Image.Image:
    """Simulate JPEG compression artefacts at low quality."""
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    result = Image.open(buf).copy()
    buf.close()
    # Restore original mode if grayscale
    if img.mode == "L":
        result = result.convert("L")
    return result


def salt_and_pepper(img: Image.Image, amount: float = 0.05) -> Image.Image:
    """Add salt-and-pepper noise (random black/white pixels)."""
    arr = np.array(img).copy()
    n_pixels = int(amount * arr.size)
    # Salt
    coords = [np.random.randint(0, d, n_pixels // 2) for d in arr.shape[:2]]
    arr[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, d, n_pixels // 2) for d in arr.shape[:2]]
    arr[coords[0], coords[1]] = 0
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Named augmentation registry
# ---------------------------------------------------------------------------

AUGMENTATIONS = {
    "Gaussian Noise (σ=25)": lambda img: add_gaussian_noise(img, std=25),
    "Gaussian Noise (σ=50)": lambda img: add_gaussian_noise(img, std=50),
    "Low Contrast (40%)": lambda img: reduce_contrast(img, factor=0.4),
    "Low Contrast (20%)": lambda img: reduce_contrast(img, factor=0.2),
    "Motion Blur": lambda img: motion_blur(img, kernel_size=9),
    "JPEG Artefacts (Q=15)": lambda img: jpeg_compress(img, quality=15),
    "Salt & Pepper (5%)": lambda img: salt_and_pepper(img, amount=0.05),
}


def get_augmentation_names() -> list[str]:
    return list(AUGMENTATIONS.keys())


def apply_augmentation(img: Image.Image, name: str) -> Image.Image:
    """
    Apply a named augmentation to a PIL Image.

    Args:
        img:  Input PIL Image.
        name: One of the keys returned by get_augmentation_names().

    Returns:
        Augmented PIL Image.

    Raises:
        ValueError if name is not recognised.
    """
    if name not in AUGMENTATIONS:
        raise ValueError(
            f"Unknown augmentation '{name}'. "
            f"Available: {list(AUGMENTATIONS.keys())}"
        )
    return AUGMENTATIONS[name](img)


def apply_random_augmentation(img: Image.Image) -> tuple[Image.Image, str]:
    """
    Apply a randomly chosen augmentation.

    Returns:
        (augmented_image, augmentation_name)
    """
    name = random.choice(list(AUGMENTATIONS.keys()))
    return apply_augmentation(img, name), name
