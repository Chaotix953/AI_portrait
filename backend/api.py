
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from PIL import Image
import torch
import numpy as np
import cv2
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from tempfile import NamedTemporaryFile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fonctions utilitaires (copiées du script original)
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


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
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

@app.post("/generate")
async def generate_image(
    content: UploadFile = File(...),
    style: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form("low quality, blurry, artifacts"),
    num_inference_steps: int = Form(40),
    guidance_scale: float = Form(6.5),
    strength: float = Form(0.55)
):
    # Sauvegarde temporaire des fichiers uploadés
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_content:
        shutil.copyfileobj(content.file, temp_content)
        content_path = temp_content.name
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_style:
        shutil.copyfileobj(style.file, temp_style)
        style_path = temp_style.name
    try:
        content_img = load_image(content_path)
        style_img = load_image(style_path)
        content_colorized = reinhard_color_transfer(content_img, style_img)
        canny = make_canny_map(content_img, low=100, high=200)
        generator = torch.Generator(device=device).manual_seed(42)
        result = pipe(
            prompt=prompt,
            image=content_colorized,
            control_image=canny,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            negative_prompt=negative_prompt,
            generator=generator,
            ip_adapter_image=style_img,
        ).images[0]
        output_path = "results/styled_image.png"
        os.makedirs("results", exist_ok=True)
        result.save(output_path)
        return FileResponse(output_path, media_type="image/png")
    finally:
        os.remove(content_path)
        os.remove(style_path)


"""
curl -X POST "http://127.0.0.1:8000/generate" \
  -F "content=@images/content.jpg" \
  -F "style=@images/style.jpg" \
  -F "prompt=portrait, oil painting" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=7.5" \
  -F "strength=0.6" \
  --output styled_image.png
"""