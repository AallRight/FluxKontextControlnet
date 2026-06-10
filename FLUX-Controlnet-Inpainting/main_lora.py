import torch
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux_kontext import FluxTransformer2DModel
from pipeline_kontext_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
import os

check_min_version("0.30.2")

# ==================== Configuration ====================
# Model paths (update these to your local paths)
FLUX_KONTEXT_PATH = "path/to/FLUX.1-Kontext-dev"
CONTROLNET_PATH = "path/to/FLUX.1-dev-Controlnet-Inpainting-Beta"
LORA_PATH = "path/to/lora_checkpoint/step_12000"  # Set to None if not using LoRA

# Input paths
image_path = "path/to/background.png"           # Background image
mask_path = "path/to/mask.png"                  # Inpaint mask
front_path = "path/to/foreground_reference.png"  # Foreground reference (white background)
prompt = ""

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

# Load LoRA weights (optional)
if LORA_PATH is not None:
    pipe.load_lora_weights(LORA_PATH)

# ==================== Load Images ====================
size = (768, 768)
image = load_image(image_path).convert("RGB").resize(size)
mask = load_image(mask_path).convert("RGB").resize(size)
front_image = load_image(front_path).convert("RGB").resize(size)

# ==================== Batch Generation ====================
seed_list = [24, 42, 100, 123, 200, 250, 300, 500]
save_dir = "inpaint_results"
os.makedirs(save_dir, exist_ok=True)

for idx, seed in enumerate(seed_list):
    print(f"\n[{idx + 1}/{len(seed_list)}] Generating with seed: {seed}")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        kontext_image=front_image,
        prompt=prompt,
        height=size[1],
        width=size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=28,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=1.0,  # Beta: 1.0, Alpha: 3.5
    ).images[0]

    save_path = os.path.join(save_dir, f"result_seed_{seed}.png")
    result.save(save_path)
    print(f"Saved: {save_path}")

print(f"\nDone! Results saved to: {save_dir}/")