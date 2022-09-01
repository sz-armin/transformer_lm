import torch

device = torch.device("cuda")

x = torch.randn(4, 1, 16).to(device=device)

encoder_layer = torch.nn.TransformerEncoderLayer(
    d_model=16,
    nhead=1,
    batch_first=True,
    dropout=0.1).to(device=device)
encoder_layer.eval()

with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.float16):
        y = encoder_layer(x)