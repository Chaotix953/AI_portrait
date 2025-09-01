# ============================================================
# SD 1.5 + ControlNet (Canny) + IP-Adapter (style par image)
# Requis :
#   pip install --upgrade diffusers transformers accelerate safetensors
#   pip install pillow opencv-python numpy scikit-image
#   # PyTorch CUDA (cu121 conseillé) + xFormers (optionnel) :
#   # pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
#   # pip install xformers
# ============================================================

import torch
from PIL import Image
import numpy as np
import cv2

from diffusers import ControlNetModel,StableDiffusionControlNetImg2ImgPipeline

def load_image(path, max_side=768):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img

def reinhard_color_transfer(content_img: Image.Image, style_img: Image.Image) -> Image.Image:
    try:
        from skimage import color
    except ImportError:
        return content_img

    c = np.asarray(content_img).astype(np.float32) / 255.0
    s = np.asarray(style_img).astype(np.float32) / 255.0

    h = min(c.shape[0], s.shape[0])
    w = min(c.shape[1], s.shape[1])
    c_res = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
    s_res = cv2.resize(s, (w, h), interpolation=cv2.INTER_AREA)

    c_lab = color.rgb2lab(c_res)
    s_lab = color.rgb2lab(s_res)

    c_mean, c_std = c_lab.reshape(-1,3).mean(0), c_lab.reshape(-1,3).std(0) + 1e-6
    s_mean, s_std = s_lab.reshape(-1,3).mean(0), s_lab.reshape(-1,3).std(0) + 1e-6
    out_lab = ((c_lab - c_mean) / c_std) * s_std + s_mean

    out_rgb = np.clip(color.lab2rgb(out_lab), 0, 1)
    out_full = cv2.resize(out_rgb, content_img.size, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray((out_full*255).astype(np.uint8))

def make_canny_map(img: Image.Image, low=100, high=200) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    edges = cv2.Canny(arr, low, high)
    edges_3 = np.stack([edges]*3, axis=-1)
    return Image.fromarray(edges_3)

# ---------- Entrées ----------
CONTENT_PATH = "images/download.jpg"   # image de contenu
STYLE_PATH   = "images/style.jpg"     # image de style (référence IP-Adapter)

content = load_image(CONTENT_PATH)
style   = load_image(STYLE_PATH)

# Optionnel : rapprocher la palette du style AVANT SD (souvent + cohérent)
content_colorized = reinhard_color_transfer(content, style)

# ControlNet : préserver la structure du contenu
canny = make_canny_map(content, low=100, high=150)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=dtype,
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype,
).to(device)

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin",
    image_encoder_folder="image_encoder",
)

pipe.set_ip_adapter_scale(0.65)

prompt = ""  # vous pouvez guider la texture: "oil painting, thick brush strokes"
negative_prompt = "low quality, blurry, artifacts"

generator = torch.Generator(device=device).manual_seed(42)

result = pipe(
    prompt=prompt,
    image=content_colorized,              # image init (après transfert de palette)
    control_image=canny,                  # structure (ControlNet Canny)
    num_inference_steps=40,
    guidance_scale=6.5,
    strength=0.55,                        # 0.4–0.65 : plus bas = plus fidèle au contenu
    negative_prompt=negative_prompt,
    generator=generator,
    ip_adapter_image=style,
).images[0]

result.save("results/styled_image.jpg")
print("✅ Image générée : styled_image.jpg")
