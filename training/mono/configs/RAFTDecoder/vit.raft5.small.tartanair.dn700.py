# Metric3D v2-S fine-tuning on TartanAir v2, run (a): depth_normalize=(0.1, 700), which covers 200 m at the
# TartanAir label scale (1000 / 308 = 3.25 at test time). Identical to run (b) otherwise.
# See reports/finetune_v2s_tartanair/PROTOCOL.md. Inference uses mono/configs/HourglassDecoder/vit.raft5.small.dn700.py.
_base_=['./vit.raft5.small.tartanair.dn200.py']

data_basic=dict(
    depth_normalize=(0.1, 700),
)
