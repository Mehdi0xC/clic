import torch
from diffusers import StableDiffusionPipeline
from utils import config_utils, debug_utils
from diffusers import DDIMScheduler
from utils import diff_utils
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils import diff_utils
from utils import pipe_utils, io_utils, debug_utils
import sys
from src.att_store import MyAttentionStore
import cv2

def main(config):

    # Env setup
    debug_utils.seed_everything(config.seed)

    if config.clear:
        io_utils.clear_dir(f"output")

    # Pipe setup
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(config.device)
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.scheduler = scheduler
    unet = pipe.unet
    def dummy_checker(images, **kwargs): return images, [False] * len(images)
    pipe.safety_checker = dummy_checker

    original_image = io_utils.load_image("input/input.png").to(config.device)

    # Attention setup
    att_store = MyAttentionStore(config, None)
    cd_weights = torch.load(f"weights/{config.weights}/pytorch_lora_weights.bin", map_location="cpu")
    custom_diffusion_attn_procs = diff_utils.get_custom_diff_attn_procs(unet, cd_weights, att_store, train_q_out=config.train_q_out)
    unet.set_attn_processor(custom_diffusion_attn_procs)
    unet.to("cuda")
    pipe.load_textual_inversion(f"weights/{config.weights}", weight_name=f"<x>.bin")

    prompt_1 = config.prompt
    # print("Prompt: ", prompt_1)

    gen = torch.Generator(device="cuda").manual_seed(config.seed)

    with torch.autocast("cuda"), torch.no_grad():
        original_latents = pipe_utils.get_latent(pipe, original_image, gen)

    # Infer setup
    gen = torch.Generator(device="cuda").manual_seed(config.seed)
    model = pipe.unet
    guidance_scale=7.5
    num_inference_steps=config.n_timesteps
    prompt_embeds_1 = pipe._encode_prompt(prompt=prompt_1, device=config.device, num_images_per_prompt=1, do_classifier_free_guidance=True)

    scheduler.set_timesteps(num_inference_steps, device=config.device)
    timesteps = scheduler.timesteps
    height = model.config.sample_size * pipe.vae_scale_factor
    width = model.config.sample_size * pipe.vae_scale_factor
    latents = None
    num_channels_latents = model.config.in_channels

    latents = pipe_utils.prepare_latents(
        pipe,
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds_1.dtype,
        gen.device,
        gen,
        latents,
    )

    latents = scheduler.scale_model_input(latents, 0)

    with torch.autocast("cuda"):

            eta = 1.0
            extra_step_kwargs = pipe_utils.prepare_extra_step_kwargs(pipe, gen, eta)

            for i, timestep in enumerate(timesteps):


                if config.do_blending:
                    print("blending")

                    with torch.no_grad():
                        latents = scheduler.add_noise(original_latents, torch.randn_like(latents), timestep)
                        torch.cuda.empty_cache()


                with torch.no_grad():
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
                    noise_pred = model(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds_1,
                    cross_attention_kwargs=None,
                    return_dict=False,
                    )[0]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
                    pipe_utils.get_image(pipe, latents).save(f"output/seg-att-{config.tag}.png")

            att_maps = []
            for key in att_store.atts.keys():
                
                for i in range(0, 50):
                
                    if "cross_up" in key:
                        att_map = att_store.atts[key][i][:,1,:,:]
                        att_map = torch.mean(att_map, dim=0).unsqueeze(0)
                        att_map = F.interpolate(att_map.unsqueeze(0).float(), size=(64, 64), mode='bilinear', align_corners=False).half()
                        att_maps.append(att_map.squeeze(0))

            att_maps = torch.cat(att_maps, dim=0)
            att_maps = att_maps.mean(dim=0).unsqueeze(0)

            mask = io_utils.save_image(att_maps, "output", f"atts_{config.seed}")
            mask = mask.numpy() if hasattr(mask, 'numpy') else np.array(mask)

            kernel = np.ones((config.dilation_factor,5),np.uint8)
            mask = cv2.erode(mask,kernel,iterations = 1)
            mask = cv2.dilate(mask,kernel,iterations = 1)

            # Threshold the mask 
            mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]

            mask_pil = Image.fromarray(mask)
            mask_pil.save(f"input/mask.png")


if __name__ == "__main__":
    config = config_utils.load_config(sys.argv)
    config.mode = "gen"
    main(config)

