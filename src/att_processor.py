
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

class MyAttentionProcessor(nn.Module):
    r"""
    Processor for implementing attention for the Custom Diffusion method.

    Args:
        train_kv (`bool`, defaults to `True`):
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `True`):
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        att_store (`AttStore`, *optional*, defaults to `None`):
            The `AttStore` object to use for manupulating the attention maps.
        att_type (`str`, *optional*, defaults to `None`):
            The type of attention to use. Can be `cross`, `self` or `None`.
        layer_name (`str`, *optional*, defaults to `None`):
            The name of the layer to use for saving the attention maps.
    """

    def __init__(
        self,
        train_kv=True,
        train_q_out=True,
        hidden_size=None,
        cross_attention_dim=None,
        out_bias=True,
        dropout=0.0,
        att_store=None,
        att_type=None, # Cross or Self or None
        layer_name = None
    ):
        super().__init__()

        print(f"Initializing ... {layer_name}")
        self.train_kv = train_kv
        self.train_q_out = train_q_out
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.att_store = att_store
        self.att_type = att_type
        self.layer_name = layer_name
        self.t = 0
        if self.att_store is not None:
            self.att_store.pred[self.layer_name] = {}
            self.att_store.null[self.layer_name] = {}
            self.att_store.cfg[self.layer_name] = {}
            self.att_store.atts[self.layer_name] = []

        self.att_store.key_save[layer_name] = []
        self.att_store.value_save[layer_name] = []

        # `_custom_diffusion` id for easy serialization and loading.
        if self.train_kv:
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_q_out:
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_custom_diffusion = nn.ModuleList([])
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if self.train_q_out:
            with torch.autocast("cuda"):
                query = self.to_q_custom_diffusion(hidden_states)
        else:
            query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.train_kv: # and "cross_up" in self.layer_name:

            key = self.to_k_custom_diffusion(encoder_hidden_states)
            value = self.to_v_custom_diffusion(encoder_hidden_states)
        else:
            with torch.autocast("cuda"):
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)    


        attention_probs = attn.get_attention_scores(query, key.half(), attention_mask)


        # Extracting the attention maps
        if self.att_type == "cross":
            if self.att_store.mode == "tune":
                self.att_store.pred[self.layer_name] = self._unravel_attn(attention_probs[:8, :, :])
                self.t += 1
            elif self.att_store.get_probs:
                att_map = attention_probs[8:, :, :].mean(0)[:, self.att_store.token].unsqueeze(0)
                if att_map.shape[1] == 4096:
                    att_map = att_map.view(64, 64)
                elif att_map.shape[1] == 1024:
                    att_map = att_map.view(32, 32)
                elif att_map.shape[1] == 256:
                    att_map = att_map.view(16, 16)
                self.att_store.cfg[self.layer_name] = att_map.unsqueeze(0)
            else:
                self.att_store.atts[self.layer_name].append(self._unravel_attn(attention_probs[8:, :, :]))
                self.t += 1

        hidden_states = torch.bmm(attention_probs, value.half())

        if self.att_store.mode == "tune":
            hidden_states = attn.batch_to_head_dim(hidden_states)
        elif self.att_store.mode == "gen":
            with torch.no_grad():
                hidden_states = attn.batch_to_head_dim(hidden_states)

        if self.train_q_out:
            with torch.autocast("cuda"):
                hidden_states = self.to_out_custom_diffusion[0](hidden_states)
                hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    # @torch.no_grad()
    def _unravel_attn(self, x):
        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            for map_ in x:
                map_ = map_.view(map_.size(0), h, w)
                maps.append(map_)

        maps = torch.stack(maps, 0) 
        return maps.permute(1, 0, 2, 3).contiguous() 
