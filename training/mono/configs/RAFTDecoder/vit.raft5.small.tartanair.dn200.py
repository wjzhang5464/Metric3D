# Metric3D v2-S fine-tuning on TartanAir v2, run (b): depth_normalize=(0.1, 200) (official setting).
# Recipe: the official KITTI fine-tuning example (vit.raft5.large.kitti.py) adapted to ViT-S, overridden by the
# project rules (20 epochs on the train split, one checkpoint per epoch). Every value and its source is listed in
# reports/finetune_v2s_tartanair/PROTOCOL.md (section 4). Run (a) is vit.raft5.small.tartanair.dn700.py.
_base_=['../_base_/losses/all_losses.py',
       '../_base_/models/encoder_decoder/dino_vit_small_reg.dpt_raft.py',
       '../_base_/datasets/tartanair_v2.py',
       '../_base_/default_runtime.py',  # missing from the official KITTI config (dist_params etc.); as in vit.raft5.small.py
       ]

model=dict(
    decode_head=dict(
        type='RAFTDepthNormalDPT5',
        iters=4,  # as the released v2-S model (the KITTI config's 8 belongs to ViT-L/g)
        n_downsample=2,
        detach=False,
    ),
)

# loss method (official KITTI fine-tuning losses)
losses=dict(
    decoder_losses=[
        dict(type='VNLoss', sample_ratio=0.2, loss_weight=0.1),
        dict(type='GRUSequenceLoss', loss_weight=1.0, loss_gamma=0.9, stereo_sup=0),
        dict(type='DeNoConsistencyLoss', loss_weight=0.001, loss_fn='CEL', scale=2)
    ],
)

data_array = [
     [
          dict(TartanAirV2='TartanAirV2_dataset'),
     ],
]

# configs of the canonical space
data_basic=dict(
    canonical_space = dict(
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.1, 200),
    crop_size = (616, 1064),  # %28 = 0
    clip_depth_range=(0.1, 200),  # missing from the official KITTI config but read by BaseDataset; test phase only
)

# 20 epochs over the 69,779 training frames at an effective batch of 16 (drop_last): 4,361 iterations per epoch
batchsize_per_gpu = 16
acc_batch = 1
thread_per_gpu = 16  # DataLoader workers; engineering setting, see PROTOCOL.md
train_frames = 69779
iters_per_epoch = train_frames // (batchsize_per_gpu * acc_batch)
epochs = 20

interval = iters_per_epoch  # one checkpoint per epoch
log_interval = 100
evaluation = dict(
    online_eval=False,  # checkpoints are selected offline on the validation split with the inference script
    interval=interval,
    metrics=['abs_rel', 'delta1', 'rmse', 'normal_mean', 'normal_rmse', 'normal_a1'],
    multi_dataset_eval=False,  # True builds a DistributedSampler, which fails with --launcher None; val loader is unused
    exclude=['DIML_indoor', 'GL3D', 'Tourism', 'MegaDepth'],
)

# save checkpoint during training, with '*_AMP' is employing the automatic mix precision training
checkpoint_config = dict(by_epoch=False, interval=interval)
runner = dict(type='IterBasedRunner_AMP', max_iters=iters_per_epoch * epochs)

# optimizer (official KITTI fine-tuning values)
optimizer = dict(
    type='AdamW',
    encoder=dict(lr=5e-7, betas=(0.9, 0.999), weight_decay=0, eps=1e-10),
    decoder=dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=0, eps=1e-10),
    strict_match = True
)
# schedule (official KITTI shape; the poly decay spans max_iters)
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=20,
                 warmup_ratio=1e-6,
                 power=0.9, min_lr=1e-8, by_epoch=False)
