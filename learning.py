# Built on top of Custom Diffusion training script from 

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import random
import shutil
import copy
from pathlib import Path
import re
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import torchvision.transforms.functional as TF
import sys
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from utils import config_utils, io_utils

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from src.att_store import MyAttentionStore
from src.att_processor import MyAttentionProcessor

# Utils import
from utils.diff_utils import get_mappings


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger(__name__)


def freeze_params(params):
    for param in params:
        param.requires_grad = False

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    rec_ids = [example["rec_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    aug_mask = [example["mask"] for example in examples]
    att_mask = [example["instance_att_mask"] for example in examples]
    class_ids = [example["class_prompt_ids"] for example in examples]

    input_ids = torch.cat(input_ids, dim=0)
    rec_ids = torch.cat(rec_ids, dim=0)
    class_ids = torch.cat(class_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    aug_mask = torch.stack(aug_mask)
    att_mask = torch.stack(att_mask)

    # convert att_mask to grayscale
    att_mask = att_mask.mean(dim=1)
    # convert to 64x64
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).half()
    aug_mask = aug_mask.to(memory_format=torch.contiguous_format).half()
    att_mask = att_mask.to(memory_format=torch.contiguous_format).half()

    rec_pixel_values = pixel_values.clone() * att_mask

    att_mask = F.interpolate(att_mask.unsqueeze(1).float(), size=(64, 64), mode="bilinear", align_corners=False).half()
    att_mask = att_mask.squeeze(0)

    context_mask = att_mask.clone() 

    # Replace zeros with 0.5 in the attention mask
    context_mask[context_mask < 0.05] = 0.25

    context_mask = context_mask * aug_mask

    batch = {"input_ids": input_ids, "rec_ids": rec_ids, "pixel_values": pixel_values, "mask": aug_mask.unsqueeze(1), "att_mask": att_mask.unsqueeze(1), "class_ids": class_ids, "rec_pixel_values": rec_pixel_values, "context_mask": context_mask.unsqueeze(1)}
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        mask_size=64,
        att_mask_size=64,
        center_crop=False,
        hflip=False,
        aug=True,
    ):
        self.size = size
        self.mask_size = mask_size
        self.att_mask_size = att_mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug
        self.instance_att_masks_path = []
        self.class_att_masks_path = []
        self.instance_images_path = []
        self.class_images_path = []
        self.color_jitter = transforms.ColorJitter(brightness=0.5)
        # self.random_grayscale = transforms.RandomGrayscale(p=0.5)
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))

        for concept in concepts_list:
            inst_img_path = [
                (x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]
            inst_att_mask_path = [
                (x, concept["instance_prompt"]) for x in Path(concept["instance_att_mask_dir"]).iterdir() if x.is_file()
            ]
            # Repeat the att_masks to match the number of instance images
            if len(inst_img_path) == 1:
                inst_img_path = inst_img_path * 5
                inst_att_mask_path = inst_att_mask_path * 5


            self.instance_images_path.extend(inst_img_path)
            self.instance_att_masks_path.extend(inst_att_mask_path)

        z = list(zip(self.instance_images_path, self.instance_att_masks_path))
        random.shuffle(z)
        self.instance_images_path, self.instance_att_masks_path = zip(*z)
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images
        self.flip = TF.hflip

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, att_mask, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        att_mask = att_mask.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        att_mask = np.array(att_mask).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.half)
        att_mask = (att_mask / 255.0).astype(np.half)


        instance_image = np.zeros((self.size, self.size, 3), dtype=np.half)
        attention_mask = np.zeros((self.size, self.size, 3), dtype=np.half)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            attention_mask = att_mask[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            attention_mask[top : top + inner, left : left + inner, :] = att_mask
            mask[
                top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 1.0
        return instance_image, mask, attention_mask

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        attention_mask, _ = self.instance_att_masks_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)#.convert("L")
        attention_mask = Image.open(attention_mask)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")            

        instance_image = self.color_jitter(instance_image)
        # instance_image = self.random_grayscale(instance_image)
        flip_flag = random.random() < 0.5
        if flip_flag:
            instance_image = self.flip(instance_image)
            attention_mask = self.flip(attention_mask)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.33
                else np.random.randint(int(1.125 * self.size), int(1.5 * self.size))
            )
        instance_image, mask, attention_mask = self.preprocess(instance_image, attention_mask, random_scale, self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        example["instance_att_mask"] = torch.from_numpy(attention_mask).permute(2, 0, 1)
        modifier_token = re.findall(r'<(.*?)>', instance_prompt)[0]
        reconstructor_prompts = [f"A photo of <x>",
                                 "A pattern of <x>",
                                 "<x>",
                                 "<x> on a blank background",
                                 "A photo of <x> on a blank background",
                                 "<x> style",
                                    "A photo of <x> style",
                                    "<x> style on a blank background",
                                ]

        class_prompt = config.class_prompt




        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids


        rec_id = random.sample(reconstructor_prompts ,1)[0].replace("<x>", f"<{modifier_token}>")
        print(rec_id)
        example["rec_prompt_ids"] = self.tokenizer(
            rec_id,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        

        example["class_prompt_ids"] = self.tokenizer(
            class_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids


        return example


def save_new_embed(text_encoder, modifier_token_id, accelerator, args, output_dir):
    """Saves the new token embeddings from the text encoder."""
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    for x, y in zip(modifier_token_id, args.modifier_token):
        learned_embeds_dict = {}
        learned_embeds_dict[y] = learned_embeds[x]
        torch.save(learned_embeds_dict, f"{output_dir}/{y}.bin")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Custom Diffusion training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_att_mask_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_mode",
        type=str,
        default="style",
        help="Mode of the training",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom-diffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--attention_lambda",
        type=float,
        default=0.333,
        help="Attention loss weight",
    )
    parser.add_argument(
        "--context_lambda",
        type=float,
        default=0.333,
        help="Context loss weight",
    )
    parser.add_argument(
        "--local_lambda",
        type=float,
        default=0.333,
        help="Local loss weight",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--freeze_model",
        type=str,
        default="crossattn_kv",
        choices=["crossattn_kv", "crossattn"],
        help="crossattn to enable fine-tuning of all params in the cross attention",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    ) # Was defaulted on 500
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-1, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default="ktn+pll+ucd", help="A token to use as initializer word."
    )
    parser.add_argument("--hflip", default=True, action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Dont apply augmentation during data augmentation when this flag is enabled.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(config, args):
    logging_dir = Path(args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("custom-diffusion", config=vars(args))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": config.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "instance_att_mask_dir": args.instance_att_mask_dir,
                "class_data_dir": args.class_data_dir,
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    initializer_token_id = []
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split("+")
        args.initializer_token = args.initializer_token.split("+")
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(
            args.modifier_token, args.initializer_token[: len(args.modifier_token)]
        ):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode([initializer_token], add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for x, y in zip(modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)
    ########################################################
    ########################################################

    vae.requires_grad_(False)
    if args.modifier_token is None:
        text_encoder.requires_grad_(False)
    unet.requires_grad_(True)
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if accelerator.mixed_precision != "fp16" and args.modifier_token is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    attention_class = MyAttentionProcessor
    # now we will add new Custom Diffusion weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
    att_mappings = get_mappings()

    train_kv = True
    train_q_out = config.train_q_out
    custom_diffusion_attn_procs = {}
    import sys
    sys.argv = ['train.py', 'default']
    config = config_utils.load_config(sys.argv)
    config.mode = "tune"
    att_store = MyAttentionStore(config, None)

    st = unet.state_dict()
    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        layer_name = name.split(".processor")[0]
        weights = {
            "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
            "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
        }
        if train_q_out:
            weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
            weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
            weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=train_kv,
                train_q_out=train_q_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                att_store = att_store,
                att_type = "cross",
                layer_name=att_mappings[name]
            ).to(unet.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
        else:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=False,
                train_q_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                att_store = att_store,
                att_type = "self",
                layer_name = att_mappings[name]
            )
    del st
    unet.set_attn_processor(custom_diffusion_attn_procs)

    custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)
    accelerator.register_for_checkpointing(custom_diffusion_layers)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.modifier_token is not None:
            text_encoder.gradient_checkpointing_enable()

    optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        itertools.chain(text_encoder.get_input_embeddings().parameters(), custom_diffusion_layers.parameters())
        if args.modifier_token is not None
        else custom_diffusion_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = CustomDiffusionDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        size=args.resolution,
        mask_size=vae.encode(
            torch.randn(1, 3, args.resolution, args.resolution).to(dtype=weight_dtype).to(accelerator.device)
        )
        .latent_dist.sample()
        .size()[-1],
        center_crop=args.center_crop,
        hflip=args.hflip,
        aug=not args.noaug,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    if args.modifier_token is not None:
        custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    k = 0
    initial_regularizer_noise = None

    for epoch in range(first_epoch, args.num_train_epochs):
        print(args.num_train_epochs)
        unet.train()
        if args.modifier_token is not None:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            k += 1
            print(global_step)

            with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                rec_latents = vae.encode(batch["rec_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                rec_latents = rec_latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noisy_rec_latents = noise_scheduler.add_noise(rec_latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                rec_hidden_states = text_encoder(batch["rec_ids"])[0]
                class_hidden_states = text_encoder(batch["class_ids"])[0]

                m_token = rec_hidden_states[0][1]
                c_token = class_hidden_states[0][1]
                text_embedding_loss = -torch.sqrt(torch.sum((m_token - c_token) ** 2))

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample


                # Locate 40408 or 49409 INDEX in the batch (to find modifier token indices after text augmentation)
                if 49408 in batch["input_ids"][0]:
                    idx = torch.where(batch["input_ids"][0] == 49408)[0].item()
                else:
                    idx = torch.where(batch["input_ids"][0] == 49409)[0].item()


                save_att = True
                if save_att:
                    attentions = []
                    class_maps = []
                    style_maps = []
                    for key in att_store.pred.keys():
                        if "cross_up" in key: # or "cross_down_2" in key or "cross_mid" in key:
                        # if config.target_layers in key:

                            att_map = att_store.pred[key][:,idx,:,:]
                            att_map = torch.mean(att_map, dim=0).unsqueeze(0)
                            att_map = F.interpolate(att_map.unsqueeze(0).float(), size=(64, 64), mode='bilinear', align_corners=False).half()
                            attentions.append(att_map.squeeze(0))


                    attentions = torch.cat(attentions, dim=0)
                    attentions = torch.mean(attentions, dim=0)
                    attentions = attentions.unsqueeze(0)




                local_pred = unet(noisy_rec_latents, timesteps, rec_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                if args.instance_mode == "style":
                    # Calculate the attention loss
                    att_mask = batch["att_mask"]
                    attentions = attentions - attentions.min()
                    attentions = attentions / (attentions.max() - attentions.min())
                    attention_loss = F.mse_loss(attentions.half(), att_mask.half(), reduction="mean")
                    attention_loss = attention_loss * args.attention_lambda

                    # Calculate the diffusion loss
                    context_mask = batch["context_mask"]
                    context_loss = F.mse_loss(model_pred.half(), target.half(), reduction="none")
                    context_loss = ((context_loss * context_mask).sum([1, 2, 3]) / (context_mask.sum([1, 2, 3])+1e-10)).mean()
                    context_loss = context_loss * args.context_lambda

                    # Calculate the reconstruction loss
                    local_mask = batch["att_mask"] # Patch reconstruction mask is the same as attention mask
                    local_loss = F.mse_loss(local_pred.half(), target.half(), reduction="none")
                    local_loss = ((local_loss * local_mask).sum([1, 2, 3]) / (local_mask.sum([1, 2, 3])+1e-10)).mean()
                    local_loss = local_loss * args.local_lambda

                    loss = attention_loss + local_loss + context_loss # + text_embedding_loss

                elif args.instance_mode == "seg":
                    
                    # Calculate the attention loss
                    att_mask = batch["att_mask"]
                    attentions = attentions - attentions.min()
                    attentions = attentions / (attentions.max() - attentions.min())
                    attention_loss = F.mse_loss(attentions.half(), att_mask.half(), reduction="mean")
                    attention_loss = attention_loss

                    loss = attention_loss


                elif args.instance_mode == "break":
                    # Calculate the attention loss
                    att_mask = batch["att_mask"]
                    attentions = attentions - attentions.min()
                    attentions = attentions / (attentions.max() - attentions.min())
                    attention_loss = F.mse_loss(attentions.half(), att_mask.half(), reduction="mean")
                    attention_loss = attention_loss * args.attention_lambda

                    # Calculate the diffusion loss
                    context_mask = batch["att_mask"]
                    context_loss = F.mse_loss(model_pred.half(), target.half(), reduction="none")
                    context_loss = ((context_loss * context_mask).sum([1, 2, 3]) / (context_mask.sum([1, 2, 3])+1e-10)).mean()
                    context_loss = context_loss * args.context_lambda

                    loss = attention_loss + context_loss

                elif args.instance_mode == "content": # E. g. just learn the content image using custom diffusion. You'd also need to clean other parts of the code for faster training.

                    # Calculate the diffusion loss
                    mask = batch["mask"]
                    diff_loss = F.mse_loss(model_pred.half(), target.half(), reduction="none")
                    diff_loss = ((diff_loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                    loss = diff_loss 

                accelerator.backward(loss)
                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if args.modifier_token is not None:
                    if accelerator.num_processes > 1:
                        grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
                    for i in range(len(modifier_token_id[1:])):
                        index_grads_to_zero = index_grads_to_zero & (
                            torch.arange(len(tokenizer)) != modifier_token_id[i]
                        )
                    grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
                        index_grads_to_zero, :
                    ].fill_(0)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(text_encoder.parameters(), custom_diffusion_layers.parameters())
                        if args.modifier_token is not None
                        else custom_diffusion_layers.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.is_main_process:

                            if (global_step+1) % config.validation_steps == 0:

                                io_utils.save_image(attentions, "output", f"{global_step}_atts")
                                io_utils.save_image(batch["pixel_values"].squeeze(0), "output", f"{global_step}_img")

                                if (global_step+1) % 100 == 0:
                                    val_text_encoder = accelerator.unwrap_model(text_encoder)
                                    val_unet = accelerator.unwrap_model(unet)

                                    pipeline = DiffusionPipeline.from_pretrained(
                                        args.pretrained_model_name_or_path,
                                        unet=val_unet,
                                        text_encoder=val_text_encoder,
                                        tokenizer=tokenizer,
                                        revision=args.revision,
                                    )
                                    
                                    pipeline.unet.to(accelerator.device)
                                    pipeline.text_encoder.to(accelerator.device)
                                    pipeline.vae.to(accelerator.device)
                                    def dummy_checker(images, **kwargs): return images, [False] * len(images)
                                    pipeline.safety_checker = dummy_checker
                                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                                    pipeline.set_progress_bar_config(disable=True)


                                    with torch.autocast("cuda"):
                                    # run inference
                                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                                        for i in range(1):
                                            print(args.instance_prompt)
                                            pipeline("<x>", num_inference_steps=50, generator=generator, eta=1.0).images[0].save(f"output/{global_step+1}_local_{i}.png")
                                            pipeline(args.instance_prompt, num_inference_steps=50, generator=generator, eta=1.0).images[0].save(f"output/{global_step+1}_global_{i}.png")
                                            # pipeline("A chair", num_inference_steps=50, generator=generator, eta=1.0).images[0].save(f"output/{global_step+1}_reg_{i}.png")

                                    del pipeline
                                    torch.cuda.empty_cache()
                                    # Save the custom diffusion layers
                                    accelerator.wait_for_everyone()
                                    if accelerator.is_main_process:
                                        unet.save_attn_procs(f"{args.output_dir}/{global_step+1}")
                                        save_new_embed(text_encoder, modifier_token_id, accelerator, args, f"{args.output_dir}/{global_step+1}")

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step > args.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    config = config_utils.load_config(sys.argv)
    sys.argv = sys.argv[1:]
    args = parse_args()

    main(config, args)