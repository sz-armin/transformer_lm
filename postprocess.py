# Converted DeepSpeed checkpoint
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

save_path = "/home/is/armin-sa/Projects/lm/data/checkpoints_dec/last-v1.ckpt"
output_path = "/home/is/armin-sa/Projects/lm/e8.ckpt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)