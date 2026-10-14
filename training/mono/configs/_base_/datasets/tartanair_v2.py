# TartanAir v2 trainval package (fine-tuning). Paths come from data_server_info/tartanair_v2.py.
# Training pipeline: the official KITTI fine-tuning pipeline (configs/RAFTDecoder/vit.raft5.large.kitti.py),
# except RandomResize is_lidar=False (nearest-neighbour label resize), which the official v2 training config
# uses for every dense dataset, including the synthetic ones (VKITTI, Hypersim, Replica).
# See reports/finetune_v2s_tartanair/PROTOCOL.md.

TartanAirV2_dataset=dict(
    lib = 'TartanAirV2Dataset',
    data_name = 'TartanAirV2',
    data_type = 'denselidar_syn',
    data = dict(
    train=dict(
        sample_ratio = 1.0,
        sample_size = -1,
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=False),
                    dict(type='RandomCrop',
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='rand',
                         ignore_label=-1,
                         padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                         mask_maxsize=50,
                         prob=0.2,
                         rgb_invalid=[0,0,0],
                         label_invalid=-1,),
                  dict(type='RandomHorizontalFlip',
                       prob=0.4),
                  dict(type='PhotoMetricDistortion',
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur',
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],),
    # Not used for checkpoint selection (that runs the inference script on the validation split);
    # the official training loop only builds it.
    val=dict(
        sample_ratio = 1.0,
        sample_size = 20,
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop',
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center',
                         ignore_label=-1,
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],),
    ),
)
