import torch
from collections import OrderedDict

def load_openai_whisper(model, ckpt):
    ckpt_list = []
    state_dict_save = OrderedDict()
    i=0 
    # breakpoint()
    for k,v in ckpt.items():
        if '.num_batches_tracked' not in k :
            ckpt_list.append(v)

    # breakpoint()       
    for k, v in model.state_dict().items():
        state_dict_save[k] = v  #原始Whisper模型的state_dict
    
    for k, v in state_dict_save.items(): 
        if 'wgt_alpha' not in k and 'act_alpha' not in k and 'running_max' not in k and 'running_num' not in k\
            and 'scales' not in k and 'smooth_vector' not in k\
            and 'running_max_c' not in k and 'running_min_c' not in k and 'running_num_c' not in k and 'act_min' not in k\
            and 'running_max_t' not in k and 'running_min_t' not in k and 'running_num_t' not in k\
            and 'sym_z' not in k and 'ot_fused' not in k and 'wgt_alpha_test' not in k\
            and 'act_alpha_qk' not in k and 'act_alpha_v' not in k and 'running_max_qk' not in k and 'running_max_v' not in k\
            and 'running_max_input' not in k and 'running_max_output'not in k and 'running_min_input' not in k and 'running_min_output' not in k\
            and 'ln_alpha' not in k and 'running_max_ln' not in k and 'running_num_ln' not in k\
            and 'act_attn_alpha' not in k and 'running_max_attn' not in k and 'running_num_attn' not in k\
            and 'act_cross_attn_alpha' not in k and 'running_max_cross_attn' not in k and 'running_num_cross_attn' not in k\
            and 'act_mlp_alpha' not in k and 'running_max_mlp' not in k and 'running_num_mlp' not in k\
            and 'act_x_alpha' not in k and 'act_pe_alpha' not in k and 'running_max_pe' not in k and 'running_num_pe' not in k and 'running_max_x' not in k and 'running_num_x' not in k\
            and 'act_cf_alpha' not in k and 'running_max_cf' not in k and 'running_num_cf' not in k\
            and 'act_te_alpha' not in k and 'running_max_te' not in k\
            and 'act_logits_alpha' not in k and 'running_max_logits' not in k and 'running_num_logits' not in k\
            and 'act_shift' not in k and 'running_min' not in k\
            and 'act_logits_alpha' not in k and 'running_max_logits' not in k and 'running_num_logits' not in k\
            and 'act_shift' not in k and 'running_min' not in k and 'wgt_zeros' not in k and 'running_mean' not in k\
            and 'act_zerop' not in k and 'lut' not in k and not 'req' in k and not 'shiftmount' in k and not 'multiplier' in k\
            and 'de_smooth_vec' not in k and 'input_alpha' not in k:
            state_dict_save[k] = ckpt_list[i] #把timm预训练模型的参数赋值给state_dict_save
            i+=1
    
    
    return model.load_state_dict(state_dict_save) 


        