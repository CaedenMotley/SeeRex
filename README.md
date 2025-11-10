# SeeRex
# DINOv3 Attention Heatmap Visualizer 

This repository provides a simple tool for visualizing **attention maps** from [Meta’s DINOv3](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) model.  
It extracts and overlays **CLS to patch attention** from the model’s internal layers to show **where the network is “looking”** when interpreting an image.

---

##  What Is an Attention Heatmap?

In transformer based vision models like DINOv3, every image is divided into small **patches**, and the model learns how each patch relates to others through a mechanism called **self attention**.  
The **CLS token** (short for “classification”) acts like a global summary; it gathers information from all patches to make predictions.

A **CLS to patch attention heatmap** shows *which image regions most strongly influenced that summary*.  
- Bright areas indicate patches the model considered *important*.  
- Dark areas indicate patches it largely *ignored*.

In short, this visualization offers a peek into **how the model perceives and prioritizes different parts of an image**.

---
## Example of Heatmap Output

Below is an example showing how the attention heatmap highlights the most influential regions of the image as seen by DINOv3.

<table align="center">
  <tr valign="top">
    <td align="center" width="50%">
      <div style="display:flex; flex-direction:column; align-items:center; justify-content:flex-start; padding:10px;">
        <img src="images/LadNCow.jpeg" alt="Original Image" width="95%" style="border-radius:10px; display:block; margin-bottom:10px;"/>
        <p style="margin:0;"><b>Original:</b> Me and a very patient Highland cow.</p>
      </div>
    </td>
    <td align="center" width="50%">
      <div style="display:flex; flex-direction:column; align-items:center; justify-content:flex-start; padding:10px;">
        <img src="data/visuals/LadNCow_overlay.png" alt="DINOv3 Attention Overlay" width="95%" style="border-radius:10px; display:block; margin-bottom:10px;"/>
        <p style="margin:0;"><b>Overlay:</b> Attention heatmap from DINOv3’s final transformer layer (head 4), showing where the model focused most strongly when forming its final image representation. Brighter regions indicate the areas that influenced it the most.</p>
      </div>
    </td>
  </tr>
</table>


##  Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dinov3-heatmap.git
   cd dinov3-heatmap
   
2. **Install dependencies**
pip install torch torchvision transformers timm matplotlib pillow opencv-python python-dotenv

3. **Request Access to DINOV3 Model**
unfortutunately you must first request access to the model on hugging face to utilize it at the moment.
The model used here:
https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m

Request Access (requires a Hugging Face account).
Wait for approval (usually within a few hours).
After approval, generate a read access token and set it as an environment variable:

```bash
HF_TOKEN="your_huggingface_token_here"
```

4. **Testing with your own image**
Place your test image in the data/ directory.
run:   python visualize.py

The script will:
* Load the DINOv3 model from Hugging Face
* Compute the CLS attention map from the final layer
* Overlay it on the image
* Display and save the result in outputs/

##  Current Limitations
* Heatmap offset bug
When saving the attention overlay as a PNG, the saved image may not perfectly match the displayed one. There is a small alignment/misalignment issue being debugged.

* Single heatmap only
Right now the script produces only one heatmap (typically from the final layer and averaged). A full visualization across all layers/heads is a work in progress.

* Static image selection
Currently the script expects one image and does not yet offer CLI arguments or a UI to pick images.

##  Work in Progress
* implementation of a **D3.js** and **JavaScript** web interface for real time,  
  interactive exploration of DINOv3 attention heatmaps within the browser.This will be worked on once current limitations are fixed.

## Attribution

This project builds on the DINOv3 model released by Meta AI.
Model: [https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
License and access: gated on Hugging Face under Meta AI terms
