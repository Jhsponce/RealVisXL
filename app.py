import gradio as gr
import torch
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers import EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torchvision.transforms as T

# ---- Load pre-trained models (.safetensors) ----
base_model_path = hf_hub_download("Scythd/RealVisV2", "realvisxlV20_v20Bakedvae.safetensors")
controlnet_path = hf_hub_download("Scythd/PromeAi-SDXL-ControlNet", "sdxl-controlnet-lineart-promeai-fp16.safetensors")

# ---- Load ControlNet ----
controlnet = ControlNetModel.from_single_file(
    controlnet_path,
    torch_dtype=torch.float16
).to("cuda")

# ---- Load SDXL pipeline ----
pipe = StableDiffusionXLControlNetPipeline.from_single_file(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# ---- Load depth estimation model ----
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

# ---- Preprocessing ----
def preprocess_sketch(sketch: Image.Image, apply_canny: bool, apply_contrast: bool, apply_depth: bool) -> Image.Image:
    sketch = sketch.convert("RGB").resize((1024, 1024))

    if apply_canny:
        sketch_np = np.array(sketch)
        gray = cv2.cvtColor(sketch_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        sketch = Image.fromarray(edges).convert("RGB")

    if apply_contrast:
        sketch = ImageEnhance.Contrast(sketch).enhance(1.5)

    if apply_depth:
        inputs = feature_extractor(images=sketch, return_tensors="pt").to("cuda")
        with torch.no_grad():
            prediction = depth_model(**inputs).predicted_depth[0]
            depth = T.Resize((1024, 1024))(prediction.unsqueeze(0)).squeeze().cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth_img = Image.fromarray((depth * 255).astype(np.uint8)).convert("RGB")
            sketch = Image.blend(sketch, depth_img, alpha=0.4)

    return sketch

# ---- Prompt weighting helper ----
def auto_weight_prompt(prompt: str) -> str:
    weighted_keywords = ["curved pathway","top-side window"
    ]
    for kw in weighted_keywords:
        if kw in prompt:
            prompt = prompt.replace(kw, f"({kw}:1.5)")
    return prompt

# ---- Core generation function ----
def generate(sketch, prompt, negative_prompt, guidance, steps, strength, seed, apply_canny, apply_contrast, apply_depth):
    if sketch is None or not prompt.strip():
        return None

    sketch = preprocess_sketch(sketch, apply_canny, apply_contrast, apply_depth)
    generator = torch.manual_seed(int(seed)) if seed > 0 else None
    prompt = auto_weight_prompt(prompt.lower())

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=sketch,
        controlnet_conditioning_scale=strength,
        guidance_scale=guidance,
        num_inference_steps=int(steps),
        generator=generator
    )
    return result.images[0]

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("## SDXL Sketch-to-Render (PromeAI Style) + Preprocessing + Auto-Weight Prompt")
    with gr.Row():
        with gr.Column():
            sketch = gr.Image(type="pil", label="Lineart Sketch")
            prompt = gr.Textbox(label="Prompt", value="Photo of a two-story house, large glass windows, stone tiles façade, second floor wood cladding, taken at noon, shot on Canon DSLR, real estate listing photo, dry concrete, weathered stone, dirt on windows, overcast lighting, rain, fog.")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="cartoon, CGI, oversharpened, stylized, unrealistic textures, perfect surfaces, additional windows")
            guidance = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="Guidance Scale")
            steps = gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
            strength = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="ControlNet Strength")
            seed = gr.Number(label="Seed (0 = random)", value=0, precision=0)
            apply_canny = gr.Checkbox(label="Apply Canny Edge Detection", value=False)
            apply_contrast = gr.Checkbox(label="Boost Contrast", value=True)
            apply_depth = gr.Checkbox(label="Overlay Depth Map", value=False)
            btn = gr.Button("Generate")
        with gr.Column():
            output = gr.Image(type="file", format="png", label="Generated Image")


    btn.click(
        fn=generate,
        inputs=[sketch, prompt, negative_prompt, guidance, steps, strength, seed, apply_canny, apply_contrast, apply_depth],
        outputs=output
    )

demo.launch()
