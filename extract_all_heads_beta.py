'''
    This file generates the head vis at each layer as well as combined heads
    Much more refined than other file but still needs documentation.
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import cv2
from PIL import Image
from dino_import import load_dinov3

# Constants
DATA_DIR = Path("data")
IMAGE_PATH = Path("images/LadNCow.jpeg").resolve()
# literally never fucking runs cuda for some reason but whatever cpu works!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (1024, 1536)
PATCH_SIZE = 16

def get_cls_attention_map(
    attentions: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    grid_h: int,
    grid_w: int,
    gamma: float = 0.5,
    percentile_clip: float = 99.9
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
    # this varies greaatly from its parent file and will be debugged as progress is made
    # this shall eventually userp the parent code!

    cls_attn_row = attentions[layer_idx][0, head_idx, 0]
    num_patches = grid_h * grid_w

    # Take the last N patches
    patch_attn = cls_attn_row[-num_patches:]

    cls_attn_map = patch_attn.reshape(grid_h, grid_w).cpu().numpy()


    # This removes the edge sinks being created causing visual bugs
    cls_attn_map[0, :] = 0  # Top row
    cls_attn_map[-1, :] = 0  # Bottom row
    cls_attn_map[:, 0] = 0  # Left col
    cls_attn_map[:, -1] = 0  # Right col

    vmax = np.percentile(cls_attn_map, percentile_clip)
    if vmax > 0:
        cls_attn_map = np.clip(cls_attn_map / vmax, 0, 1)

    cls_attn_map = np.power(cls_attn_map, gamma)

    return cls_attn_map

def overlay_attention(image: np.ndarray, attn_map: np.ndarray, alpha=0.45):
    """
    Overlays attention heatmap on the original image.
    """
    attn_resized = cv2.resize(attn_map, (image.shape[1], image.shape[0]))

    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_INFERNO)

    #  Convert BGR - >  RGB
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)

    return overlay
def save_all_animations(image_np, attentions, grid_h, grid_w, folder_name):
    """Generates individual attn head GIFs and one grid GIF."""

    out_dir = DATA_DIR / "visuals" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]


    print(f"Generating {num_heads} individual GIFs...")
    for h in range(num_heads):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        ims = []
        for l in range(num_layers):
            m = get_cls_attention_map(attentions, l, h, grid_h, grid_w)
            ov = overlay_attention(image_np, m)
            im = ax.imshow(ov, animated=True)
            txt = ax.text(0.55, 0.95, f"Head {h} - Layer {l}", transform=ax.transAxes,
                          ha="center", color="white", fontsize=12)
            ims.append([im, txt])

        ani = animation.ArtistAnimation(fig, ims, interval=400, blit=True)
        ani.save(out_dir / f"head_{h}_evolution.gif", writer='pillow')
        plt.close()

    print("Generating Collective Grid GIF...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.02, hspace=0.1)
    axes = axes.flatten()
    grid_ims = []

    for l in range(num_layers):
        step_artists = []
        for h in range(num_heads):
            m = get_cls_attention_map(attentions, l, h, grid_h, grid_w)
            ov = overlay_attention(image_np, m)
            im = axes[h].imshow(ov, animated=True)
            axes[h].axis('off')
            if l == 0: axes[h].set_title(f"Head {h}", color='black')
            step_artists.append(im)

        # Add a simple global layer counter
        title = fig.text(0.5, 0.92, f"DINOv3 Layer {l}",
                         ha="center", fontsize=16, weight='bold')
        step_artists.append(title)
        grid_ims.append(step_artists)

    ani_grid = animation.ArtistAnimation(fig, grid_ims, interval=500, blit=False)
    ani_grid.save(out_dir / "all_heads_grid_evolution.gif", writer='pillow')
    plt.close()
    print(f"Done! Check {out_dir}")


def save_layer_evolution_gif(
        image_rgb: np.ndarray,
        layer_maps: list[np.ndarray],
        output_path: Path,
        fps: int = 4
):
    """
    Saves a single GIF showing the averaged attention
    evolving from Layer 0 to the final layer.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')

    ims = []
    for i, attn_map in enumerate(layer_maps):

        overlay_rgb = overlay_attention(image_rgb, attn_map)

        im = ax.imshow(overlay_rgb, animated=True)
        # Simple title showing just the layer number
        title = ax.text(0.5, 1.02, f"Layer {i}", transform=ax.transAxes,
                        ha="center", fontsize=14, fontweight='bold')
        ims.append([im, title])

    ani = animation.ArtistAnimation(fig, ims, interval=750 / fps, blit=True)
    ani.save(output_path, writer='pillow')
    plt.close()
    print(f"Layer evolution GIF saved to {output_path}")


def extract_and_visualize(image_path: Path):

    # Load and resize the image (control scale here)
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize(TARGET_SIZE)
    image_np = np.array(image_resized)

    grid_h, grid_w = TARGET_SIZE[1] // PATCH_SIZE, TARGET_SIZE[0] // PATCH_SIZE

    print("Loading model...")
    processor, model = load_dinov3()
    model.to(DEVICE)

    # Preprocess (disable internal resize i already handle it)
    inputs = processor(images=image_resized, return_tensors="pt", do_resize=False).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    total_tokens = outputs.attentions[0].shape[-1]
    expected_patches = grid_h * grid_w
    diff = total_tokens - expected_patches

    print(f"--- DIAGNOSTIC SHIZ ---")
    print(f"Total tokens in sequence: {total_tokens}")
    print(f"Expected patch tokens: {expected_patches}")
    print(f"Extra tokens found: {diff}")

    # Saves individual heads and the 2 x 3 grid
    save_all_animations(image_np, outputs.attentions, grid_h, grid_w, image_path.stem)

    #averages each head into 1
    num_layers = len(outputs.attentions)
    layer_maps = []
    num_patches = grid_h * grid_w  # Calculate this once

    print("Generating Averaged Layer Evolution...")
    for l in range(num_layers):
        # Average across all heads for this layer
        # attentions[l][0] is (Heads, Tokens, Tokens) -> mean(0) is (Tokens, Tokens)
        avg_attn_matrix = outputs.attentions[l][0].mean(dim=0)

        # Get the CLS token's attention row (index 0)
        avg_cls_row = avg_attn_matrix[0]

        # take the last num_patches tokens
        patch_attn = avg_cls_row[-num_patches:]

        # Reshape to image dimensions
        cls_map = patch_attn.reshape(grid_h, grid_w).cpu().numpy()

        # Normalize and apply brightness correction
        if cls_map.max() > 0:
            cls_map /= cls_map.max()
        else:
            cls_map = np.zeros((grid_h, grid_w), dtype=np.float32)

        layer_maps.append(np.power(cls_map, 0.5))

    out_gif = DATA_DIR / "visuals" / image_path.stem / f"{image_path.stem}_evolution.gif"
    save_layer_evolution_gif(image_np, layer_maps, out_gif)

if __name__ == "__main__":
    extract_and_visualize(IMAGE_PATH)