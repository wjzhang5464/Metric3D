# Supplementary material draft: Metric3D v2 baseline on TartanAir v2

Draft text for the paper's supplementary material. Every statement is backed by
`reports/finetune_v2s_tartanair/PROTOCOL.md` (sources, decision log, measurements). Numbers in brackets
point to its sections.

---

## Metric3D v2 baseline

**Model.** We use Metric3D v2 with the ViT-S backbone (DINOv2-reg encoder and RAFT-style GRU decoder
with 4 update iterations), initialised from the official pretrained checkpoint
(`metric_depth_vit_small_800k.pth`). Metric3D predicts depth in a canonical camera space with focal
length 1000 px. Ground-truth labels are scaled by 1000 / f during training (the paper's CSTM_label), and
predictions are mapped back to metric depth with the inverse factor at test time. The network does not
receive the intrinsics as input; it is given the ground-truth intrinsics only for this conversion. TartanAir
does not appear in Metric3D's training data, so the zero-shot setting is genuinely zero-shot.

**Zero-shot evaluation.** We run the official inference pipeline unchanged: the official config and
weights. The 640x640 image is resized to 616x616, padded with the ImageNet mean to the 616x1064 input,
and the prediction is cropped and resized back to 640x640. We verified that our wrapper reproduces the
official test script bit for bit.

**Fine-tuning recipe.** The official repository provides one fine-tuning example (KITTI, ViT-L/g), and we
follow it, adapted to ViT-S. The paper's main text and supplement describe fine-tuning differently (20K
vs. 6K steps, with an additional pseudo-label loss that the released code does not implement), so we use
the released code as the reference. Following our common protocol for all baselines, we train for 20
epochs on the TartanAir v2 training split only (69,779 frames) and keep one checkpoint per epoch. The
settings are listed in Table S1. The whole network is trained, including the encoder, with a 20x lower
learning rate for the encoder than for the decoder. The losses are the official fine-tuning losses: an
L1 loss on the canonical depth of every GRU iteration (weight 1.0), the virtual-normal loss (0.1) and the
depth-normal consistency loss (0.001). TartanAir provides no ground-truth normals, so the normal branch
receives no direct supervision. Pixels marked invalid by the dataset's mask, i.e. sky and depth outside
(0.001, 200] m, are excluded from all supervised losses.

**Depth range.** The decoder clamps canonical depth to `depth_normalize = (0.1, 200)`. With TartanAir's
intrinsics (f = 320 px, 640x640; at test time the effective focal length after resizing is 308 px), this
caps metric depth at about 62 m, whereas the evaluation range reaches 200 m. We therefore fine-tune two
variants, both with the canonical focal length of 1000:
- (a) `depth_normalize = (0.1, 700)`, covering 200 m (test-time cap 215.6 m);
- (b) the official `(0.1, 200)` (cap 61.6 m).

Because the decoder's depth bins and update parametrisation depend on this range, variant (a) does not
reproduce the pretrained predictions at initialisation; it adapts during fine-tuning.

**Model selection.** For each variant we choose the checkpoint with the lowest mean per-frame depth MAE
on the validation split (1,944 frames), using the official metric definition and the same inference
pipeline as at test time. The test set was not used for any choice. The released training code deletes
all but the 9 most recent checkpoints. We noticed this only after training, so epochs 1-11 were
unavailable and selection was among epochs 12-20. Before looking at any result, we fixed the rule that
the selection would be rejected and training repeated if the best epoch were the earliest available one
(12). In both variants the validation MAE decreased almost monotonically from epoch 12 to epoch 19, and
epoch 19 was selected (validation MAE: (a) 1.382 m, (b) 1.658 m; the pretrained model scored 6.434 m
under (a)'s depth range and 6.149 m under the official one).

**Test protocol.** All three models (zero-shot, (a), (b)) are run on the 10,868 test frames with the
same inference script, the ground-truth intrinsics and no scale alignment. We save the de-canonicalised
metric depth at 640x640 before the official post-processing step that zeroes depths outside (0, 300) m;
this step never changed a pixel in our runs. Metrics are computed with the test package's official
evaluation script.

### Table S1. Metric3D v2-S fine-tuning settings

| Setting | Value |
|---|---|
| Initialisation | official `metric_depth_vit_small_800k.pth` |
| Trainable parameters | all (encoder and decoder) |
| Optimizer | AdamW, betas (0.9, 0.999), weight decay 0, eps 1e-10 |
| Learning rate | encoder 5e-7, decoder 1e-5 |
| Schedule | linear warm-up (20 iterations), then polynomial decay (power 0.9) to 1e-8 over the full schedule |
| Batch size | 16 (one GPU) |
| Training length | 20 epochs = 87,220 iterations (4,361 per epoch) |
| Precision | bfloat16 autocast; gradient-norm clipping 2.5 |
| Losses | GRU-sequence L1 on canonical depth (1.0, gamma 0.9), virtual-normal (0.1), depth-normal consistency (0.001) |
| Canonical focal length | 1000 px |
| `depth_normalize` | (a) (0.1, 700); (b) (0.1, 200) |
| Input crop | 616 x 1064, random crop (zero padding where the image is smaller) |
| Augmentation | official: random resize 0.85-1.15 (p = 0.5), random edge mask, horizontal flip (p = 0.4), photometric distortion, weather, blur, JPEG compression |
| Invalid pixels | dataset valid mask (sky and depth outside (0.001, 200] m excluded) |
| Model selection | lowest validation MAE (epochs 12-20; see text); epoch 19 for both variants |
| Seed | 3047 |
| Hardware / time | 1x NVIDIA A100 80GB; about 24 h per variant |

**Implementation notes (all behaviour-preserving).** The released fine-tuning config does not run as
published, so we made six changes that do not alter what is trained:
1. added the missing `clip_depth_range` and runtime defaults to the config;
2. built the unused validation loader non-distributed;
3. stopped the training loop at the last iteration (it otherwise continues until the polynomial schedule
   turns complex);
4. skipped the ImageNet initialisation file that is overwritten by the full checkpoint anyway;
5. used one OpenCV thread per data-loader worker;
6. kept all checkpoints.

For random resizing we use nearest-neighbour label resizing, as the official configuration does for all
dense datasets, instead of the KITTI example's scatter resizing for sparse LiDAR.
