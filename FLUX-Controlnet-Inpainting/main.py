import torch
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux_kontext import FluxTransformer2DModel
from pipeline_kontext_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

check_min_version("0.30.2")

# ==================== Configuration ====================
# Model paths (update these to your local paths)
FLUX_KONTEXT_PATH = "path/to/FLUX.1-Kontext-dev"
CONTROLNET_PATH = "path/to/FLUX.1-dev-Controlnet-Inpainting-Beta"

# Input paths
image_path = "path/to/background.png"          # Background image
mask_path = "path/to/mask.png"                 # Inpaint mask
front_path = "path/to/foreground_reference.png" # Foreground reference (white background)

prompt = "A puppy is sitting next to a kitty."

# ==================== Build Pipeline ====================
controlnet = FluxControlNetModel.from_pretrained(
    CONTROLNET_PATH, torch_dtype=torch.bfloat16
)
transformer = FluxTransformer2DModel.from_pretrained(
    FLUX_KONTEXT_PATH, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    FLUX_KONTEXT_PATH,
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)

# ==================== Load Images ====================
size = (768, 768)
image = load_image(image_path).convert("RGB").resize(size)
mask = load_image(mask_path).convert("RGB").resize(size)
front_image = load_image(front_path).convert("RGB").resize(size)
generator = torch.Generator(device="cuda").manual_seed(24)

# ==================== Inpaint ====================
result = pipe(
    kontext_image=front_image,
    prompt=prompt,
    height=size[1],
    width=size[0],
    control_image=image,
    control_mask=mask,
    num_inference_steps=28,
    generator=generator,
    controlnet_conditioning_scale=0.9,   # Recommended: 0.9–1.0
    guidance_scale=3.5,
    negative_prompt="",
    true_guidance_scale=1.0,             # Beta: 1.0, Alpha: 3.5
).images[0]

result.save("flux_kontext_inpaint.png")
print("Successfully inpaint image")
