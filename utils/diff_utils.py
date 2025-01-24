from src.att_processor import MyAttentionProcessor

def get_mappings():
    key_mapping = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": "cross_down_0_0",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor": "cross_down_0_1",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor": "cross_down_1_0",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor": "cross_down_1_1",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor": "cross_down_2_0",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor": "cross_down_2_1",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor": "cross_up_1_0",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor": "cross_up_1_1",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor": "cross_up_1_2",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor": "cross_up_2_0",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor": "cross_up_2_1",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor": "cross_up_2_2",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor": "cross_up_3_0",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor": "cross_up_3_1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor": "cross_up_3_2",
        "mid_block.attentions.0.transformer_blocks.0.attn2.processor": "cross_mid_0_0",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor": "self_down_0_0",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor": "self_down_0_1",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor": "self_down_1_0",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor": "self_down_1_1",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor": "self_down_2_0",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor": "self_down_2_1",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor": "self_up_1_0",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor": "self_up_1_1",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor": "self_up_1_2",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor": "self_up_2_0",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor": "self_up_2_1",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor": "self_up_2_2",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor": "self_up_3_0",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor": "self_up_3_1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor": "self_up_3_2",
        "mid_block.attentions.0.transformer_blocks.0.attn1.processor": "self_mid_0_0"
        }
    return key_mapping

def get_custom_diff_attn_procs(unet, cd_weights, att_store, train_q_out):
    att_mappings = get_mappings()
    attention_class = MyAttentionProcessor
    custom_diffusion_attn_procs = {}
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
                train_kv=True,
                train_q_out=train_q_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                att_store = att_store,
                att_type = "cross",
                layer_name = att_mappings[name]
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
    for k in cd_weights.keys():
        print(k)
    for key in custom_diffusion_attn_procs.keys():
        print(key)
        if key in cd_weights:
            custom_diffusion_attn_procs[key].load_state_dict(cd_weights[key])
        if key + '.to_k_custom_diffusion.weight' in cd_weights:
            custom_diffusion_attn_procs[key].to_k_custom_diffusion.weight.data = cd_weights[key + '.to_k_custom_diffusion.weight']#.half()
        if key + '.to_v_custom_diffusion.weight' in cd_weights:
            custom_diffusion_attn_procs[key].to_v_custom_diffusion.weight.data = cd_weights[key + '.to_v_custom_diffusion.weight']#.half()
        if key + '.to_q_custom_diffusion.weight' in cd_weights:
            custom_diffusion_attn_procs[key].to_q_custom_diffusion.weight.data = cd_weights[key + '.to_q_custom_diffusion.weight']#.half()
        if key + '.to_out_custom_diffusion.0.weight' in cd_weights:
            custom_diffusion_attn_procs[key].to_out_custom_diffusion[0].weight.data = cd_weights[key + '.to_out_custom_diffusion.0.weight']#.half()
        if key + '.to_out_custom_diffusion.0.bias' in cd_weights:
            custom_diffusion_attn_procs[key].to_out_custom_diffusion[0].bias.data = cd_weights[key + '.to_out_custom_diffusion.0.bias']#.half()
    return custom_diffusion_attn_procs


