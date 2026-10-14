import csv
import io
import os
import zipfile

import cv2
import numpy as np

from .__base_dataset__ import BaseDataset


class TartanAirV2Dataset(BaseDataset):
    """TartanAir v2 trainval package (`/depot/chan129/data/TartanAir_v2/tartanair_v2_trainval`).

    Frames are read straight from the package zips; nothing is extracted. The frame list is the package's
    `frames.tsv`, filtered by its `role` column (train phase -> 'train', val phase -> 'validation'), so no
    annotation json is needed. Each zip member directory `<frame_key>/` holds rgb.png, depth.npy (float16,
    planar Z in metres), valid_mask.png and sky_mask.png (1-bit PNGs; cv2 decodes them as 0/255).

    Depth labels are invalid (-1) wherever valid_mask is 0, i.e. depth <= 1e-3, depth > 200 m or sky (the
    package's rule). Sky pixels are also given the semantic id 142, which the official code uses for sky.
    There are no GT normals (all zeros, as in BaseDataset).
    """
    ROLE_OF_PHASE = {'train': 'train', 'val': 'validation'}
    SKY_ID = 142

    def __init__(self, cfg, phase, **kwargs):
        self._zips = {}
        self._zips_pid = None
        super(TartanAirV2Dataset, self).__init__(cfg=cfg, phase=phase, **kwargs)

    def load_annotations(self):
        if self.phase not in self.ROLE_OF_PHASE:
            raise ValueError(f'TartanAirV2Dataset supports phases {list(self.ROLE_OF_PHASE)}, got {self.phase}')
        if not os.path.exists(self.data_annos_path):
            raise RuntimeError(f'Cannot find {self.data_annos_path}.')
        role = self.ROLE_OF_PHASE[self.phase]
        with open(self.data_annos_path, newline='') as f:
            rows = [r for r in csv.DictReader(f, delimiter='\t') if r['role'] == role]
        files = [dict(rgb=r['frame_key'] + '.png',  # only used to name outputs
                      zip=r['zip'],
                      frame_key=r['frame_key'],
                      cam_in=[float(r['fx']), float(r['fy']), float(r['cx']), float(r['cy'])])
                 for r in rows]
        return {'files': files}

    def _zip(self, name):
        # Zip handles must not be shared across DataLoader worker processes: reopen after a fork.
        if self._zips_pid != os.getpid():
            self._zips = {}
            self._zips_pid = os.getpid()
        if name not in self._zips:
            self._zips[name] = zipfile.ZipFile(os.path.join(self.data_root, name))
        return self._zips[name]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_zips'] = {}
        state['_zips_pid'] = None
        return state

    def load_data_path(self, meta_data):
        return dict(zip=meta_data['zip'], frame_key=meta_data['frame_key'])

    def load_batch(self, meta_data, data_path):
        zf = self._zip(data_path['zip'])
        key = data_path['frame_key']
        # BGR uint8, as cv2.imread in BaseDataset.load_data (the pipeline starts with BGR2RGB)
        curr_rgb = cv2.imdecode(np.frombuffer(zf.read(key + '/rgb.png'), np.uint8), cv2.IMREAD_COLOR)
        # float64: RandomResize asserts labels are np.float
        curr_depth = np.load(io.BytesIO(zf.read(key + '/depth.npy'))).astype(np.float64)
        valid = cv2.imdecode(np.frombuffer(zf.read(key + '/valid_mask.png'), np.uint8), cv2.IMREAD_UNCHANGED) > 0
        sky = cv2.imdecode(np.frombuffer(zf.read(key + '/sky_mask.png'), np.uint8), cv2.IMREAD_UNCHANGED) > 0
        if curr_rgb is None or curr_rgb.shape[:2] != curr_depth.shape or valid.shape != curr_depth.shape:
            raise RuntimeError(f'{data_path["zip"]}:{key} has inconsistent members.')
        curr_depth[~valid] = -1

        H, W = curr_depth.shape
        curr_sem = np.full((H, W), -1, dtype=np.int64)
        curr_sem[sky] = self.SKY_ID

        curr_intrinsic = meta_data['cam_in']
        return dict(
            curr_rgb=curr_rgb,
            curr_depth=curr_depth,
            curr_sem=curr_sem,
            curr_normal=self.load_norm_label(None, H=H, W=W),
            curr_cam_model=self.create_cam_model(H, W, curr_intrinsic),
            curr_stereo_depth=self.load_stereo_depth_label(None, H=H, W=W),
        )
