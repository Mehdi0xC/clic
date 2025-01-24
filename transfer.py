import torch
from diffusers import StableDiffusionPipeline
from utils import config_utils, debug_utils
from diffusers import DDIMScheduler
from utils import diff_utils
import torch
import torch.nn.functional as F
from utils import diff_utils
from utils import pipe_utils, io_utils, debug_utils
import sys
from src.att_store import MyAttentionStore
import torch.nn as nn

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

    # Load mask and image
    mask = io_utils.load_mask(f"input/mask.png").to(config.device)
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
    prompt_2 = "a necklace"

    gen = torch.Generator(device="cuda").manual_seed(config.seed)

    with torch.autocast("cuda"), torch.no_grad():
        original_latents = pipe_utils.get_latent(pipe, original_image, gen)

    # Infer setup
    model = pipe.unet
    guidance_scale=7.5
    num_inference_steps=config.n_timesteps
    prompt_embeds_1 = pipe._encode_prompt(prompt=prompt_1, device=config.device, num_images_per_prompt=1, do_classifier_free_guidance=True)
    prompt_embeds_2 = pipe._encode_prompt(prompt=io_utils.strip_style(prompt_1), device=config.device, num_images_per_prompt=1, do_classifier_free_guidance=True)
    prompt = prompt_embeds_1

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

    # Infer loop
    with torch.autocast("cuda"):

            eta = 1.0
            extra_step_kwargs = pipe_utils.prepare_extra_step_kwargs(pipe, gen, eta)

            t_end = 35
            t_start = config.t_start
            for i, timestep in enumerate(timesteps):


                gen_mask = mask if i < 40 else mask

                if config.do_blending:
                    print("blending")

                    with torch.no_grad():
                        
                        if 0 <= i <= 45:
                            noisy_plus = scheduler.add_noise(original_latents, torch.randn_like(latents), timestep)

                            if i < config.t_start:
                                latents = noisy_plus
                            else:
                                latents = noisy_plus * (1-gen_mask) + latents * (gen_mask)
                            del noisy_plus
                            torch.cuda.empty_cache()


                if config.do_guidance:

                    if i<30:
                        latents.requires_grad = True
                        if config.do_guidance:
                            lr = 0.001*config.att_opt if config.att_opt > 0 else 0.001*(-config.att_opt)
                            optimizer = torch.optim.Adam([latents], lr=lr)
                        else:
                            optimizer = torch.optim.Adam([latents], lr=0.0)
                        MSE = nn.MSELoss()
                        latent_model_input = torch.cat([latents] * 2)
                        att_store.get_probs = True
                        p = model(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds_1,
                        cross_attention_kwargs=None,
                        return_dict=False,
                        )[0]

                        attentions = []
                        for key in att_store.cfg.keys():
                            if key.startswith("cross_up"):
                                # Upsample attentions
                                attention = att_store.cfg[key]
                                attention = F.interpolate(attention.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False).squeeze(0)
                                attentions.append(attention)

                        attentions = torch.stack(attentions).mean(0)
                        bed_mask = torch.ones_like(attentions)
                        loss = MSE(attentions, bed_mask) if config.att_opt > 0 else -MSE(attentions, bed_mask)
                        loss.backward(retain_graph=True)
                        optimizer.step()

                latents.requires_grad = False
                att_store.get_probs = False


                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
                if i == 45 and config.do_harmonization:
                    del pipe
                    del model

                    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(config.device)
                    pipe.scheduler = scheduler
                    model = pipe.unet
                    prompt = prompt_embeds_2

                with torch.no_grad():
                    noise_pred = model(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt,
                    cross_attention_kwargs=None,
                    return_dict=False,
                    )[0]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
                    pipe_utils.get_image(pipe, latents).save(f"output/edit-{config.tag}.png")


                print(f"Step {i+1}/{len(timesteps)}")


if __name__ == "__main__":
    config = config_utils.load_config(sys.argv)
    config.mode = "gen"
    main(config)

