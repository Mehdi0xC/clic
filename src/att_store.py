
import torch.nn.functional as F

class MyAttentionStore:
    def __init__(self, config, gen_mask):
        self.mode = config.mode
        self.token = config.token_id
        self.maps = {}
        self.t_start = config.t_start
        self.t_end = config.t_end
        self.pred = {}
        self.null = {}
        self.cfg = {}
        self.atts = {}
        self.target_layers = config.target_layers
        self.key_save = {}
        self.value_save = {}
        self.key_load = {}
        self.value_load = {}
        self.probs = {}
        self.timesteps = config.n_timesteps
        self.gen_mask = gen_mask
        self.get_probs = False
    