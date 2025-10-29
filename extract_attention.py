import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dino_import import load_dinov3
import cv2
from PIL import Image


'''
 this will be reworked to allow for varying images to be selected
 for now only select the test image with given constants.
'''
DATA_DIR = Path("data")
IMAGE_PATH = Path("images/LadNCow.jpeg").resolve()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Choose how large to resize for inference (multiple of 16)
# current test image 4,000 by 6,000
TARGET_SIZE = (1024, 1536)  # preserves detail but fits in GPU/CPU memory
PATCH_SIZE = 16


def get_cls_attention_map(
    attentions: torch.Tensor,
    grid_h: int,
    grid_w: int,
    head: int = 4,
    gamma: float = 0.5,
    percentile_clip: float = 90.0
) -> np.ndarray:
    """
    Extracts and reshapes CLS to patch attention into an image space heatmap.
    Handles extra special tokens automatically and enhances contrast for visualization.

    Args:
        attentions (torch.Tensor): Model attention outputs (list of tensors per layer).
        grid_h (int): Number of patch rows.
        grid_w (int): Number of patch columns.
        head (int, optional): Which attention head to visualize. Defaults to 4.
        gamma (float, optional): Gamma correction for brightness (lower -> brighter).
        percentile_clip (float, optional): Percentile cutoff for top attention normalization.

    Returns:
        np.ndarray: Normalized 2D CLS attention heatmap.
    """

    # Extract a specific head's CLS attention
    if not isinstance(attentions, (list, tuple)) or len(attentions) == 0:
        raise ValueError("`attentions` must be a non-empty list/tuple of attention tensors.")

    if head < 0 or head >= attentions[-1].shape[1]:
        raise ValueError(f"Head index {head} out of range for available heads {attentions[-1].shape[1]}")

    # Select the last layer and chosen head
    attn = attentions[-1][0, head]  # (tokens, tokens)
    cls_attn = attn[0, 1:]          # CLS -> all other tokens

    # Validate token count and trim extra special tokens
    num_patches = grid_h * grid_w
    if cls_attn.shape[0] < num_patches:
        raise ValueError(
            f"Attention has fewer tokens ({cls_attn.shape[0]}) "
            f"than expected grid ({grid_h}×{grid_w}={num_patches}). "
            "Check model patch size or image resize transform."
        )

    # Trim any extra tokens (distillation/global tokens)
    cls_attn = cls_attn[:num_patches]

    # Reshape and normalize attention
    cls_attn_map = cls_attn.reshape(grid_h, grid_w).cpu().numpy()

    if cls_attn_map.max() == 0:
        raise ValueError("Attention map contains only zeros — invalid or empty attention values.")

    cls_attn_map /= cls_attn_map.max()

    # Apply visual enhancements
    # Gamma correction brightens contrast (gamma < 1)
    cls_attn_map = np.power(cls_attn_map, gamma)

    # Clip to top percentile to emphasize high-attention regions
    p = np.percentile(cls_attn_map, percentile_clip)
    if p > 0:
        cls_attn_map = np.clip(cls_attn_map / p, 0, 1)

    return cls_attn_map



def overlay_attention(image: np.ndarray, attn_map: np.ndarray, alpha=0.45):
    """
    Overlays attention heatmap on the original image.
    """
    attn_resized = cv2.resize(attn_map, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def save_results(attn_map: np.ndarray, overlay_rgb: np.ndarray, base_name: str):
    """
    Saves the attention map (.npy) and overlay (.png) to disk.
    """
    # currently only displaying 1 att map but will update to see all
    out_dir = DATA_DIR / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{base_name}_attention.npy", attn_map)
    plt.imsave(out_dir / f"{base_name}_overlay.png", overlay_rgb.astype(np.uint8))
    print(f" Saved attention and overlay to {out_dir}")



def extract_and_visualize(image_path: Path):
    """
    Loads DINOv3, extracts CLS-to-patch attention, visualizes and saves it.
    """
    # Load and resize the image (control scale here)
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize(TARGET_SIZE, Image.BILINEAR)
    image_np = np.array(image_resized)

    # Compute patch grid
    grid_h, grid_w = TARGET_SIZE[1] // PATCH_SIZE, TARGET_SIZE[0] // PATCH_SIZE

    # Load model
    processor, model = load_dinov3()
    model.to(DEVICE)

    # Preprocess (disable internal resize i already handle it)
    inputs = processor(images=image_resized, return_tensors="pt", do_resize=False).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract CLS attention
    attentions = outputs.attentions
    cls_map = get_cls_attention_map(attentions, grid_h, grid_w)

    # Overlay on resized image
    overlay_bgr = overlay_attention(image_np, cls_map)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Resized Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_rgb)
    plt.title("DINOv3 CLS Attention Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Save both maps
    save_results(cls_map, overlay_rgb, image_path.stem)


if __name__ == "__main__":
    extract_and_visualize(IMAGE_PATH)
