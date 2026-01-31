import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dino_import import load_dinov3
import cv2
from PIL import Image
from matplotlib.widgets import Slider



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
    head: int = 5,
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
        print("remember index starts at 0 you can not do 6 if available heads is 6")
        raise ValueError(f"Head index {head} out of range for available heads {attentions[-1].shape[1]}")

    # Select the last layer and chosen head
    attn = attentions[-1][0, head]  # (tokens, tokens)

    ''' *IMPORTANT BUG FIX* DinoV3 has 4 reg tokens'''
    NUM_REGISTER_TOKENS = 4  # DINOv3 specific

    cls_attn = attn[0, 1 + NUM_REGISTER_TOKENS:]

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
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_INFERNO) # may swap colormap

    #cv2 blends the heatmap with the orignal image (dst = alpha * src1 + beta * src2 + gamma)
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

def save_all_heads(
    image_rgb: np.ndarray,
    head_maps: list[np.ndarray],
    base_name: str
):
    out_dir = DATA_DIR / "visuals" / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for i, attn_map in enumerate(head_maps):
        overlay_bgr = overlay_attention(image_bgr, attn_map)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        np.save(out_dir / f"head_{i:02d}.npy", attn_map)
        plt.imsave(out_dir / f"head_{i:02d}.png", overlay_rgb)

    print(f"Saved {len(head_maps)} heads to {out_dir}")

def visualize_heads_interactively(
    image_rgb: np.ndarray,
    head_maps: list[np.ndarray]
):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    fig, ax = plt.subplots(figsize=(6, 8))
    plt.subplots_adjust(bottom=0.2)

    # Initial head
    current_head = 0
    overlay_bgr = overlay_attention(image_bgr, head_maps[current_head])
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    img_display = ax.imshow(overlay_rgb)
    ax.set_title(f"Attention Head {current_head}")
    ax.axis("off")

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="Head",
        valmin=0,
        valmax=len(head_maps) - 1,
        valinit=0,
        valstep=1
    )

    def update(val):
        head = int(slider.val)
        overlay_bgr = overlay_attention(image_bgr, head_maps[head])
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        img_display.set_data(overlay_rgb)
        ax.set_title(f"Attention Head {head}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def get_cls_attention_map_for_layer(
    attentions: list,
    layer_idx: int,
    grid_h: int,
    grid_w: int,
    gamma: float = 0.5,
    percentile_clip: float = 90.0
) -> np.ndarray:
    """
    Averages CLS-to-patch attention across all heads for a given layer.
    """
    layer_attn = attentions[layer_idx]  # (B, H, T, T)
    avg_attn = layer_attn[0].mean(dim=0)  # (T, T)

    NUM_REGISTER_TOKENS = 4
    cls_attn = avg_attn[0, 1 + NUM_REGISTER_TOKENS:]

    cls_attn = cls_attn[: grid_h * grid_w]
    cls_map = cls_attn.reshape(grid_h, grid_w).cpu().numpy()

    cls_map /= cls_map.max()
    cls_map = np.power(cls_map, gamma)

    p = np.percentile(cls_map, percentile_clip)
    if p > 0:
        cls_map = np.clip(cls_map / p, 0, 1)

    return cls_map

def extract_and_visualize(image_path: Path):
    """
    Loads DINOv3, extracts CLS-to-patch attention, visualizes and saves it.
    """
    # Load and resize the image (control scale here)
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize(TARGET_SIZE)
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

    '''deprecated code below for singular head '''
    # Extract CLS attention
    # attentions = outputs.attentions
    # cls_map = get_cls_attention_map(attentions, grid_h, grid_w)

    attentions = outputs.attentions
    num_heads = attentions[-1].shape[1]
    print(f"Number of attention heads: {num_heads}")

    ''' the below loop produces attn heads at final layer
    all_head_maps = []

    for head_idx in range(num_heads):
        cls_map = get_cls_attention_map(
            attentions,
            grid_h,
            grid_w,
            head=head_idx
        )
        all_head_maps.append(cls_map)'''
    num_layers = len(attentions)
    layer_maps = []

    for layer_idx in range(num_layers):
        cls_map = get_cls_attention_map_for_layer(
            attentions,
            layer_idx,
            grid_h,
            grid_w
        )
        layer_maps.append(cls_map)

    save_all_heads(image_np, layer_maps, f"{image_path.stem}_layers")
    visualize_heads_interactively(image_np, layer_maps)

'''
    Generates the side by side for GIT readme visualization
    
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
'''

if __name__ == "__main__":
    extract_and_visualize(IMAGE_PATH)
