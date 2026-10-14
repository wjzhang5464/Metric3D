db_info={}


#### TartanAir v2 trainval package (read-only). Frames are read from the package zips; the frame list is
#### frames.tsv, split by its `role` column (see mono/datasets/tartanair_v2_dataset.py).
db_info['TartanAirV2']={
    'db_root': '/depot/chan129/data/TartanAir_v2',
    'data_root': 'tartanair_v2_trainval',
    'train_annotations_path': 'tartanair_v2_trainval/frames.tsv',
    'val_annotations_path': 'tartanair_v2_trainval/frames.tsv',
}
