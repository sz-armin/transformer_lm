# Convert DeepSpeed checkpoint
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

save_path = "data/checkpoints_dec/epoch=8-step=130806.ckpt"
output_path = "e9.ckpt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)