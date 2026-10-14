# Inference config for the TartanAir v2 fine-tuned run (a): the official v2-S config with
# depth_normalize=(0.1, 700), matching training/mono/configs/RAFTDecoder/vit.raft5.small.tartanair.dn700.py.
# depth_normalize sets the decoder's depth bins and output parametrisation, so it must match training.
_base_=['./vit.raft5.small.py']

max_value = 700
data_basic=dict(
    depth_normalize=(0.1, max_value),
)
