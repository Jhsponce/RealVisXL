---
title: RealVisXL
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.36.2
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# SDXL Sketch-to-Render: Prompt Engineering

## Model and Methodology Details

### Approach: Prompt Engineering (no fine-tuning)

This project relies entirely on prompt engineering to guide the output of a pretrained Stable Diffusion XL (SDXL) model enhanced with a lineart-based ControlNet, not in fine-tune model weights. Instead, photorealistic rendering from sketches was achieved through a combination of text-based conditioning, sketch preprocessing, and parameter tuning at inference. Swerving text-to-image models towards yielding particular results can be quite challenging given the fact that randomness is still an ihnerent factor to any AI tool. 

### Model Components

## Note: All models used are publicly hosted on Hugging Face Hub, and will be automatically downloaded on first run using `hf_hub_download`.

* Base Model: `realvisxlV20_v20Bakedvae.safetensors` from `Scythd/RealVisV2` (SDXL with baked VAE) 
* ControlNet: `sdxl-controlnet-lineart-promeai-fp16.safetensors` from `Scythd/PromeAi-SDXL-ControlNet`
* Preprocessing Enhancers:

  * Canny edge detection (optional)
  * Contrast boosting
  * Depth overlay using `Intel/dpt-hybrid-midas`

### Manual Sketch Enhancement

Before inference, process some sketches were manually processed using Photoshop to improve line clarity and ControlNet responsiveness. Enhancements include:

* Inverting the sketch (white lines on black background)
* Applying the Photocopy filter to boost edges and suppress noise
* Applying a Halftone Pattern filter to simulate tonal depth in architectural shading

## Prompt Engineering Strategy

### Construction Method

It begins with a structural prompt describing the scene in plain language. For example:

```
A modern two-story house with large glass windows and a wooden facade.
```

Next step is appendding features that are often ignored or difficult for the model to capture, such as:

```
A curved stone pathway leading to the entrance, sidewalk and narrow road in front, and a small window on the top-right floor.
```

To reinforce these details, Stable Diffusion's prompt weighting syntax was applied:

```
A (modern house:1.2) with a (curved pathway:1.4), (sidewalk:1.2), and (top-right window:1.5).
```

Lastly an automated keyword weighting for common architectural features using Python string replacement logic inside the inference script was implemented.

### Iterative Refinement

During testing, the following values must be adjusted:

* ControlNet strength (0.4–0.6)
* Guidance scale (6–8.5)
* Seed value for output reproducibility 

This iterative process allows to balance prompt fidelity with structural accuracy from the sketch.

## Multimodal Strategy

In this phase the sketch is responsible for geometry and structure, while the text controls style, materials, lighting, and specific object emphasis.

* The image is passed through ControlNet’s lineart encoder.
* The prompt is passed through SDXL’s text encoder.

Together, these provide a hybrid conditioning mechanism. Prompt embeddings or external latent manipulations weren't used.

## Preprocessing Pipeline

To improve ControlNet response and prompt adherence, all sketch inputs were preprocessed and upscaled to 1024x1024 RGB

Optional preprocessing includes:

* Canny edge detection (cv2.Canny)
* Contrast enhancement (PIL.ImageEnhance)
* Depth estimation overlay (normalized depth map blended at \~0.4 alpha)

Users can toggle these in the interface.

Contrast enhancement is useful on most architectural sketches. Canny is most effective for wireframe inputs. Depth overlay adds value when the sketch contains tonal information or shading.

# The following are improvements that couldn't be implemented due to time constraints:

## Prompt Chunking (Planned Extension)

A potential enhancement would be implementing a prompt chunking system to mimic the behavior of advanced pipelines. This would allow prompts exceeding the 77-token CLIP limit to be split into logical blocks. Each block would be tokenized and selectively included based on priority and token budget.

Prompt chunks could look like:

```
Chunk 1: Base structure
Chunk 2: Spatial layout
Chunk 3: Architectural elements
Chunk 4: Style and render descriptors
```

An algorithm would select the most relevant blocks to construct a prompt within token limits. Overflow blocks could be rotated or batch-injected during reruns.

## Realism Enhancement (Planned Extension)

1. Postprocess with Blur and Grain

AI images are usually:

Too crisp

Too uniform

The app could use PIL or OpenCV to:

-Add lens blur (not Gaussian, real lens blur)

-Add a subtle RGB film grain

-Adjust levels like shadows/highlights

This simulates a real camera response.

2. Stack Inference

Once the render is complete the app could:

Pass the output through Real-ESRGAN or 4x-UltraSharp

Use CodeFormer or GFPGAN

3. SDXL Refiner

Using the SDXL 1.0 base + SDXL Refiner two-stage pipeline:

-Generates latent with the base
-Then refines detail

It improves photoreal texture, depth, and noise distribution.

## Achieving Fidelity

### Challenges

* Preserving curved pathways and the top-right small window
* Porperly rendering the wooden cladding
* Getting ControlNet to honor road/sidewalk geometry
* Preventing hallucinated stylization unrelated to the sketch

### Partial Solutions 

* Weighted prompt tokens to emphasize weak elements
* Manual sketch cleanup with filters (Photocopy, Halftone)
* Depth overlays to reinforce planar boundaries
* ControlNet strength tuning with sketch-specific adjustments


## Best Results (Reference Section)

This section summarizes values, sketches, and prompt patterns that yielded the most reliable architectural fidelity.

* Prompts:
  A.: Photo of a two-story house, large glass windows, stone tiles façade, second floor wood cladding, taken at noon, shot on Canon DSLR, real estate listing photo, dry concrete, weathered stone, dirt on windows, overcast lighting, rain, fog

  B.: A modern two-story house with stone tiles, dirty sidewalk, metal railing, and old curb. Taken on DSLR with natural overcast light, handheld photo, subtle chromatic aberration, low contrast, 35mm film


  Negative prompts: 
  
  *N1.: cartoon, CGI, over sharpened, stylized, unrealistic textures, perfect surfaces, additional windows

  *N2.: blurry, cartoonish, distorted, low resolution, sketch, grass, timber, additional windows, additional rooms, extra proportions, different textures

## All the prompts are interchangeable. Seasons and seasonal elements can be passed: winter, snow, autumn, leaves, etc.

### The preprocessed sketches can be found inside the '/sketches' folder.


* Sketch Preprocessing:
  * Inverted version of the sketch
  ![alt text](<sketchs/inverse sketch.png>)
  * Contrast Boosting: Off
  * Canny Edge Detection: Off (use for low-contrast linework)
  * Depth Overlay: Off, works best with shaded or tonal drawings

* Tunable Parameters:
  * Guidance Scale: 6-7.5
  * Inference Steps: 30
  * ControlNet Strength: 0.55-0.6
  * Seeds: 42, 200 (reproducible base)

* Prompts combination: A + N1, B + N1  

## Those yielded the following renditions:

![alt text](<outputs/image (8).png>) 
![alt text](<outputs/image (9).png>) 
![alt text](<outputs/image (10).png>) 
![alt text](<outputs/image (11).png>) 
![alt text](<outputs/image (12).png>)
![alt text](<outputs/image (15).png>) 
![alt text](<outputs/image (14).png>)



### Setup Instructions

Clone the repository:

```
git clone https://huggingface.co/spaces/Scythd/RealVisXL
pip install -r requirements.txt
python app.py
```

Or launch in browser from:

```
[text](https://huggingface.co/spaces/Scythd/RealVisXL)
```
*Select "Duplicate this Space"

```
 *For a direct run, the GPUs must be activated on request 
```


### GitHub Link

[https://github.com/Jhsponce/RealVisXL.git]


