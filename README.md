# equiSELD

SO(3) and O(3) group-equivariant attention networks for sound event detection
and localization (SELD) on first-order Ambisonics (FOA).


| `params['model']` | Model | Symmetry |
|---|---|---|
| `equiseld` | Scalar/vector Set Transformer (`equiseld.py`) | exact O(3) by construction; SO(3)-only variant available |
| `cgseld` | Clebsch-Gordan spherical-harmonic CNN, reimplementation of Sato et al. (`cgseld.py`) | SO(3) via SH tensor products |
| `seldnet` | DCASE baseline CRNN (`seldnet_model.py`) | none; optionally trained with rotation augmentation |

## Models

### equiseld: equivariant Set Transformer

`equiseld.py` works directly on FOA
waveforms.

- **Front end** (`FoaFeatures`): STFT of the ACN-ordered channels
  `[W, Y, Z, X]`, mel-banded into per-frame invariant scalars (omni and
  dipole energies, intensity norms, active/reactive cosine) and equivariant
  3-vectors (energy-normalised active and reactive intensity).
  
- **Symmetry switches** in `SeldConfig`: `equivariant=False` builds a
  parameter-matched non-equivariant twin (same shapes, per-component
  mixing and biases added); `so3_only=True` adds the cross product of the two
  intensity vectors as an axial channel, which breaks reflection
  equivariance and leaves only SO(3).
- **Losses**: `MultiAccdoaAdpitLoss` (ADPIT with the baseline's 13
  permutation hypotheses), `SingleAccdoaLoss`, and `sed_bce_aux`, an
  assignment-free detection warm-up.

`equiseld_dcase.py` adapts the model to the DCASE pipeline: it maps the
baseline parameter dict to `SeldConfig`, extracts the 11-channel features
(5 scalars + 2 vectors x 3) into the baseline's feature directory layout,
wraps the network in the `SeldModel` call signature, and provides the
optimiser recipe (AdamW with weight-decay exclusions, warmup + cosine
schedule).

### cgseld: Clebsch-Gordan network

`cgseld.py` reimplements the Sato et al. trunk in pure PyTorch: 
complex STFT of FOA is lifted to spherical-harmonic fragments
(`wxyz_stft_to_sh_input`). We replaced head by a Multi-ACCDOA head. `cgseld_dcase.py` provides the
STFT feature extraction, the pipeline wrapper, an AdamW recipe, and
`OriginalHeadLoss` (BCE + masked angular error) for the published objective.

### Baseline CRNN with rotation augmentation

For the non-equivariant baseline, three augmentations are available
(train-time only; validation and test use the stock generator):

- `acs_aug` (`acs_augmentation.py`): the 16 axis swaps and sign flips of
  the FOA channels (ACS-16), applied to the denormalised mel + intensity
  features and the labels.
- `cov_aug` (`cov_augmentation.py`): continuous O(3) augmentation. A
  rotation-complete 10-channel covariance feature (omni power, the 3x3
  dipole covariance, intensity vector) is extracted once; a fresh Haar
  O(3) element is drawn per chunk per epoch and the baseline's 7-channel
  features are assembled from the rotated covariance.
- `wave_aug` (`cls_waveform_generator.py`): the rotation is applied to the
  FOA waveform and features are recomputed on the fly.


## Setup

```bash
pip install -r requirements.txt
```

Smoke-test the standalone models:

```bash
python equiseld.py   # prints parameter count and output shapes on random audio
```

## Data

The pipeline follows the DCASE SELD baseline conventions. It expects
`<dataset_dir>/foa_dev/` (and optionally `foa_eval/`) with the `metadata_dev/`
labels, and infers the fold split and class count from the year contained in
`dataset_dir` (2020, 2021, 2022 or 2023). The defaults in `parameters.py`
point at STARSS23 (24 kHz, 13 classes, train folds 1-3, test fold 4) and at
paths on the cluster used for the experiments; change `dataset_dir`,
`feat_label_dir`, `model_dir` and `dcase_output_dir` for your machine.

Feature extraction writes one `.npy` per recording. The equivariant models
need their own feature roots because their features differ from the
baseline's;

- `equiseld`: 11 channels, unnormalised (per-component z-scoring would break
  equivariance). `parameters.py` suffixes the feature root with `_equiseld`.
- `cgseld`: 8 real channels x 1024 bins of W-phase-aligned complex STFT,
  with a per-bin scale fitted on the training folds.
- `cov_aug`: 10-channel covariance features under `cov_feat_label_dir`.


## TAU-NIGENS 2021: getting the data and training equiseld

### Download

TAU-NIGENS Spatial Sound Events 2021 is the DCASE 2021 Task 3 dataset
(600 one-minute FOA recordings at 24 kHz, 12 classes, 6 folds). It is
hosted on Zenodo: <https://zenodo.org/record/4844825> (linked from the
[DCASE 2021 Task 3 page](https://dcase.community/challenge2021/task-sound-event-localization-and-detection)).
You need the FOA development audio and its metadata.

```bash
mkdir -p /path/to/TAU2021 && cd /path/to/TAU2021
# foa_dev is a two-part zip: fetch foa_dev.zip and foa_dev.z01
# plus metadata_dev.zip from the Zenodo record, then
zip -s 0 foa_dev.zip --out foa_dev_full.zip
unzip foa_dev_full.zip
unzip metadata_dev.zip
```

After extraction the directory must look like

```
TAU2021/
  foa_dev/
    dev-train/fold1_room1_mix001.wav ...   (folds 1-4)
    dev-val/  fold5_...wav
    dev-test/ fold6_...wav
  metadata_dev/
    dev-train/fold1_room1_mix001.csv ...
    dev-val/ ...
    dev-test/ ...
```

The feature extractor walks every sub-folder of `foa_dev/` and
`metadata_dev/`, and the fold of a recording is read from the fifth
character of its file name (`fold1_...`), so the sub-folder names
themselves do not matter. The directory name must contain `2021`: that is
how `parameters.py` and `train_seldnet.py` pick 12 classes and the
DCASE 2021 protocol (train folds 1-4, validation fold 5, test fold 6).
The audio is 24 kHz, which matches the default `fs`.

### Configure

Task `40` is the reference configuration for equiseld on this dataset. Its
paths point at the cluster used for the experiments, so either edit them in
place or copy the branch under a new id:

```python
elif argv == '40':
    print("FOA + multi ACCDOA + equiseld, TAU-NIGENS 2021, DCASE2021 protocol\n")
    params['quick_test'] = False
    params['dataset'] = 'foa'
    params['multi_accdoa'] = True
    params['model'] = 'equiseld'
    params['dataset_dir'] = '/path/to/TAU2021'          # must contain "2021"
    params['feat_label_dir'] = '/path/to/feat_tau21'     # "_equiseld" is appended automatically
    params['model_dir'] = '/path/to/models'
    params['dcase_output_dir'] = '/path/to/results'
    params['nb_epochs'] = 100
```


### Extract features and train

```bash
# one-off: equiseld front-end features (11 channels, unnormalised) + ADPIT labels
python batch_feature_extraction.py 40

# train on folds 1-4, select the checkpoint on fold 5, report fold 6
python train_seldnet.py 40 <job-id>
```

The run prints per-epoch validation ER / F / LE / LR / SELD score and, at
the end, the fold-6 test scores with jackknife confidence intervals. The
best checkpoint is saved as
`<model_dir>/40_<job-id>_dev_split0_multiaccdoa_foa_model.h5`. To train on a
subset of folds for data-scaling experiments, set
`params['train_splits'] = [1, 2]` and so on; validation and test folds stay
fixed. To confirm the trained model is equivariant on real features, run
`python verify_equivariance.py 40:<job-id>` (needs the helper modules noted



Built-in task ids include:

| id | configuration |
|---|---|
| `1` | quick test with default parameters |
| `2`, `3` | baseline CRNN, FOA, single / multi ACCDOA |
| `4`-`7` | baseline CRNN on MIC input (GCC or SALSA-Lite) |
| `8`, `9` | equiseld, multi / single ACCDOA |
| `19`, `20` | cgseld |
| `12`, `23` | baseline CRNN, parameter-matched to equiseld / stock size |
| `16`-`18`, `21`, `24` | baseline CRNN with ACS / covariance / waveform augmentation |


Checkpoints go to `model_dir` as `<task>_<job>_dev_split0_<accdoa|multiaccdoa>_foa_model.h5`

### Checking equivariance

```bash
python verify_equivariance.py <task-id>:<job-id> [...]
```

For a trained checkpoint, this applies a random proper rotation and a random
reflection to the input intensity vectors and compares the output ACCDOA
vectors against the rotated reference. The O(3) model should sit at float32
round-off for both; an SO(3)-only model only for the proper rotation.

## Known limitations

- The equivariant models consume FOA only; MIC inputs are supported for the
  baseline CRNN alone.

## License

MIT, see `LICENSE`. The vendored `group-equiv-seld/` directory is the
reference implementation of Sato, Niwa and Kobayashi, "Ambisonic Signal
Processing DNNs Guaranteeing Rotation, Scale and Time Translation
Equivariance" (IEEE/ACM TASLP, 2021), and is distributed under its own
license; `group-equiv-seld/fCGModule.py` originates from
[zlin7/CGNet](https://github.com/zlin7/CGNet) (MIT).
