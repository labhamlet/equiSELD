# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.

def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs
    
        finetune_mode = False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='models/1_1_foa_dev_split6_model.h5',

        dataset_dir='/projects/0/prjs1261/seld/STARSS2023',

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir='/projects/0/prjs1261/seld/STARSS2023/seld_feat_label',
 
        model_dir='/projects/0/prjs1261/seld/STARSS2023/models',            # Dumps the trained models and training curves in this folder
        dcase_output_dir='/projects/0/prjs1261/seld/STARSS2023/results/',    # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,   # official 24 kHz STARSS23 (verified; replaced the earlier
                    # 32 kHz resampled copy on 2026-08-26)

        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 2000,
        fmax_spectra_salsalite = 9000,

        # MODEL TYPE
        model='seldnet',     # 'seldnet' - baseline CRNN or 'equiseld' - SO(3)-equivariant Set-Transformer (equiseld.py)
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=50,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        self_attn=True,
        nb_heads=8,
        nb_self_attn_layers=2,
        
        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,              # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average='macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,
        sed_threshold=0.5,      # ACCDOA activity gate at decode time
                                # (owner decision 2026-08-27: keep 0.5)
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '8':
        print("FOA + multi ACCDOA + equiseld\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'

    elif argv == '9':
        print("FOA + single ACCDOA + equiseld\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False
        params['model'] = 'equiseld'

    elif argv == '10':
        print("FOA + multi ACCDOA + equiseld, train folds [2,3] (data-fraction)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['train_splits'] = [2, 3]

    elif argv == '11':
        print("FOA + multi ACCDOA + equiseld, train fold [3] only (data-fraction)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['train_splits'] = [3]

    elif argv == '12':
        print("FOA + multi ACCDOA + SELDNet CRNN, parameter-matched to equiseld\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        # sized to match equiseld task 8: CRNN 2,252,085 params vs
        # equiseld 2,252,877 (-0.04%), with nb_cnn2d_filt=64 unchanged
        params['rnn_size'] = 240
        params['fnn_size'] = 384
        # baseline features live on scratch: the prjs1261 quota is full
        # (recomputable; note scratch-shared auto-purges after ~14 days)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'

    elif argv == '13':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched), train fold [3] only\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['train_splits'] = [3]
        # baseline features live on scratch (see task 12 note)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'

    elif argv == '14':
        print("FOA + multi ACCDOA + equiseld, FRONTAL-restricted train (|az|<=90), full-sphere test\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        # restricted split built by make_restricted_split.py --src-task 8 --dst-task 14
        # (the '_equiseld' suffix is appended below)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_frontal'
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10

    elif argv == '15':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched), FRONTAL-restricted train, full-sphere test\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        # restricted split built by make_restricted_split.py --src-task 12 --dst-task 15
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_frontal'
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10

    elif argv == '16':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched) + ACS-16 augmentation, full data\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['acs_aug'] = True      # 16 FOA rotations/reflections, train-time
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'

    elif argv == '17':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched) + continuous O(3) aug + ACS-16, full data\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['acs_aug'] = True      # dynamic ACS-16 on top of static O(3) copies
        # originals (symlinks) + K Haar-rotated copies, built by
        # make_rotated_copies.py --src-task 12 --dst-task 17
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_o3'

    elif argv == '18':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched) + on-the-fly waveform O(3) aug, full data\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['wave_aug'] = True     # fresh Haar O(3) per chunk per epoch
        params['wave_num_workers'] = 12
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'

    elif argv == '19':
        print("FOA + multi ACCDOA + cgseld (Sato et al. reimpl., our recipe), full data\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        # complex-STFT features: 1024 freq bins, 8 rows (4 ch x Re/Im)
        params['nb_mel_bins'] = 1024
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_cgseld'
        # our training recipe (owner decision; original used Adam 1e-3 x300ep)
        params['batch_size'] = 8    # 16 OOMs the 40GB A100 (verified);
        params['lr'] = 3e-4         # complex activations; 32 risks A100 OOM
        params['weight_decay'] = 0.05
        params['warmup_steps'] = 2000
        params['grad_clip'] = 1.0

    elif argv == '20':
        print("FOA + multi ACCDOA + cgseld (Sato et al. reimpl.), train fold [3] only\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_cgseld'
        params['batch_size'] = 8    # see task 19 memory note
        params['lr'] = 3e-4
        params['weight_decay'] = 0.05
        params['warmup_steps'] = 2000
        params['grad_clip'] = 1.0
        params['train_splits'] = [3]

    elif argv == '21':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched) + continuous O(3) covariance-feature aug, full data\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['cov_aug'] = True      # fresh Haar O(3) per chunk per epoch,
        params['cov_num_workers'] = 8 # applied on rotation-complete features
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'
        params['cov_feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_cov'

    elif argv == '22':
        print("FOA + multi ACCDOA + equiseld, full data, batch 128\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_batch'] = 128   # batch ablation vs task 8's 32

    elif argv == '23':
        print("FOA + multi ACCDOA + SELDNet CRNN, STOCK size (official baseline config), full data\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        # everything at defaults: rnn 128 / fnn 128 (~0.75M), Adam 1e-3,
        # batch 128, no schedule — only the feature dir moved to scratch
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'

    elif argv == '24':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched) + waveform O(3) aug, train fold [3] only\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['wave_aug'] = True     # fresh Haar O(3) per chunk per epoch
        params['wave_num_workers'] = 12
        params['train_splits'] = [3]
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'

    elif argv == '25':
        print("equiseld fold[3], lr 3e-4, 300 epochs (stretched cosine)\n")
        params.update(quick_test=False, dataset='foa', multi_accdoa=True, model='equiseld')
        params['train_splits'] = [3]; params['nb_epochs'] = 300; params['eval_every_n_epochs'] = 10; params['equiseld_batch'] = 128; params['equiseld_warmup'] = 500

    elif argv == '26':
        print("equiseld fold[3], lr 1e-3, 100 epochs\n")
        params.update(quick_test=False, dataset='foa', multi_accdoa=True, model='equiseld')
        params['train_splits'] = [3]; params['equiseld_lr'] = 1e-3; params['eval_every_n_epochs'] = 10; params['equiseld_batch'] = 128; params['equiseld_warmup'] = 500

    elif argv == '27':
        print("equiseld fold[3], lr 5e-4, 200 epochs\n")
        params.update(quick_test=False, dataset='foa', multi_accdoa=True, model='equiseld')
        params['train_splits'] = [3]; params['equiseld_lr'] = 5e-4; params['nb_epochs'] = 200; params['eval_every_n_epochs'] = 10; params['equiseld_batch'] = 128; params['equiseld_warmup'] = 500

    elif argv == '28':
        print("equiseld fold[3], lr 3e-4, 200 epochs, constant lr after warmup\n")
        params.update(quick_test=False, dataset='foa', multi_accdoa=True, model='equiseld')
        params['train_splits'] = [3]; params['nb_epochs'] = 200; params['lr_floor'] = 1.0; params['eval_every_n_epochs'] = 10; params['equiseld_batch'] = 128; params['equiseld_warmup'] = 500

    elif argv == '29':
        print("FOA + multi ACCDOA + equiseld fold[3], constant-lr recipe (arm 4) at 100 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['train_splits'] = [3]; params['lr_floor'] = 1.0; params['eval_every_n_epochs'] = 10; params['equiseld_batch'] = 128; params['equiseld_warmup'] = 500

    elif argv == '30':
        print("FOA + multi ACCDOA + stock SELDNet (published recipe), train fold [3], 200 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        # stock model sizes and recipe (rnn 128 / fnn 128, Adam 1e-3)
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'

    elif argv == '31':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched), train fold [3], 200 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'

    elif argv == '32':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched) + waveform O(3) aug, train fold [3], 200 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['wave_aug'] = True     # fresh Haar O(3) per chunk per epoch
        params['wave_num_workers'] = 12
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'

    elif argv == '33':
        print("FOA + multi ACCDOA + equiseld, FRONTAL fold [3] only, full-sphere test, 200 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        # built by make_restricted_split.py --src-task 8 --dst-task 33 --train-folds 3
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_frontal_f3'

    elif argv == '34':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched), FRONTAL fold [3] only, full-sphere test, 200 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        # built by make_restricted_split.py --src-task 12 --dst-task 34 --train-folds 3
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_frontal_f3'

    elif argv == '35':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched), full data [1,2,3], 200 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'

    elif argv == '36':
        print("FOA + multi ACCDOA + equiseld, full data [1,2,3], 200 epochs (const-lr recipe)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        # arm-4 recipe validated on fold [3] (task 28)
        params['equiseld_batch'] = 128
        params['equiseld_warmup'] = 500
        params['lr_floor'] = 1.0

    elif argv == '37':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched) + waveform O(3) aug, full data [1,2,3], 200 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['wave_aug'] = True     # fresh Haar O(3) per chunk per epoch
        params['wave_num_workers'] = 12
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k'

    elif argv == '38':
        print("FOA + multi ACCDOA + cgseld (Sato et al. reimpl.), PUBLISHED recipe, fold [3], 200 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_cgseld'
        # published recipe (TASLP 2021): plain Adam lr 1e-3, wd 1e-5,
        # batch 12, sequence ~128 STFT frames (label_seq 26 -> 130), their
        # 300 epochs capped at 200 (owner request); no warmup/cosine/clip
        params['original_recipe'] = True
        params['disable_tf32'] = True   # full fp32: TF32 spikes destabilize CG products
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10

    elif argv == '39':
        print("FOA + multi ACCDOA + equiseld, full data [1,2,3], 200 epochs (standard cosine recipe)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        # native recipe as task 8 (batch 32, lr 3e-4, warmup 2000, cosine
        # to 1% floor) -- only the horizon differs; recipe-attribution
        # control for the const-lr full-data run (task 36)

    elif argv == '40':
        print("FOA + multi ACCDOA + equiseld, TAU-NIGENS 2021, DCASE2021 protocol (train 1-4, val 5, test 6)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        # native cosine recipe, validation every epoch
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '41':
        print("FOA + multi ACCDOA + cgseld (published recipe), TAU-NIGENS 2021, fold [1] only, val every epoch\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21_cgseld'
        # published recipe as task 38; fold-1-only trainability probe
        params['original_recipe'] = True
        params['disable_tf32'] = True   # full fp32: TF32 spikes destabilize CG products
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['train_splits'] = [1]
        params['nb_epochs'] = 100

    elif argv == '42':
        print("FOA + multi ACCDOA + equiseld, TAU-NIGENS 2021, fold [1] only, val every epoch\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '43':
        print("FOA + multi ACCDOA + equiseld, TAU-NIGENS 2021, train folds [1,2] (data scaling)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1, 2]
        params['nb_epochs'] = 100
        # native cosine recipe, validation every epoch (protocol of task 40)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '44':
        print("FOA + multi ACCDOA + equiseld, TAU-NIGENS 2021, train folds [1,2,3] (data scaling)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1, 2, 3]
        params['nb_epochs'] = 100
        # native cosine recipe, validation every epoch (protocol of task 40)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '45':
        print("FOA + multi ACCDOA + equiseld 0.74M (stock-SELDNet-matched), TAU-NIGENS 2021, train [1,2,3,4] (model scaling)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['equiseld_d_s'] = 72
        params['equiseld_d_v'] = 20
        params['nb_epochs'] = 100
        # native cosine recipe, validation every epoch (protocol of task 40)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '46':
        print("FOA + multi ACCDOA + equiseld 5.91M (cgseld-matched), TAU-NIGENS 2021, train [1,2,3,4] (model scaling)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['equiseld_d_s'] = 208
        params['equiseld_d_v'] = 52
        params['nb_epochs'] = 100
        # native cosine recipe, validation every epoch (protocol of task 40)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '47':
        print("FOA + multi ACCDOA + equiseld 10.1M, TAU-NIGENS 2021, train [1,2,3,4] (model scaling)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['equiseld_d_s'] = 272
        params['equiseld_d_v'] = 68
        params['nb_epochs'] = 100
        # native cosine recipe, validation every epoch (protocol of task 40)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '48':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched) + waveform O(3) aug, TAU-NIGENS 2021, train [1] (data scaling)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['wave_aug'] = True     # fresh Haar O(3) per chunk per epoch
        params['wave_num_workers'] = 12
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1]
        params['nb_epochs'] = 100
        # stock CRNN recipe, validation every epoch (protocol of tasks 42-44)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '49':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched) + waveform O(3) aug, TAU-NIGENS 2021, train [1, 2] (data scaling)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['wave_aug'] = True     # fresh Haar O(3) per chunk per epoch
        params['wave_num_workers'] = 12
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1, 2]
        params['nb_epochs'] = 100
        # stock CRNN recipe, validation every epoch (protocol of tasks 42-44)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '50':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched) + waveform O(3) aug, TAU-NIGENS 2021, train [1, 2, 3] (data scaling)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['wave_aug'] = True     # fresh Haar O(3) per chunk per epoch
        params['wave_num_workers'] = 12
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1, 2, 3]
        params['nb_epochs'] = 100
        # stock CRNN recipe, validation every epoch (protocol of tasks 42-44)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '51':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched) + waveform O(3) aug, TAU-NIGENS 2021, train [1, 2, 3, 4] (data scaling)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['wave_aug'] = True     # fresh Haar O(3) per chunk per epoch
        params['wave_num_workers'] = 12
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1, 2, 3, 4]
        params['nb_epochs'] = 100
        # stock CRNN recipe, validation every epoch (protocol of tasks 42-44)
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '52':
        print("FOA + multi ACCDOA + stock SELDNet CRNN, TAU-NIGENS 2021, train [1,2,3,4]\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        # stock model (rnn 128 / fnn 128) and recipe, validation every epoch
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '53':
        print("FOA + multi ACCDOA + SELDNet CRNN (param-matched), TAU-NIGENS 2021, train [1,2,3,4]\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['rnn_size'] = 240      # matched to equiseld (see task 12)
        params['fnn_size'] = 384
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        # stock recipe, validation every epoch
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '54':
        print("FOA + single ACCDOA + stock SELDNet CRNN, TAU-NIGENS 2021, train [1,2,3,4]\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False   # single-ACCDOA, as the published DCASE2021 baseline
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        # stock model (rnn 128 / fnn 128) and recipe, validation every epoch
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '55':
        print("FOA + multi ACCDOA + stock SELDNet CRNN + waveform O(3) aug, TAU-NIGENS 2021, train [1,2,3,4]\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['wave_aug'] = True     # fresh Haar O(3) per chunk per epoch
        params['wave_num_workers'] = 12
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        # stock model (rnn 128 / fnn 128) and recipe, validation every epoch
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '56':
        print("FOA + multi ACCDOA + cgseld (published recipe), TAU-NIGENS 2021, train [1,2,3,4]\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21_cgseld'
        # published recipe as tasks 38/41; validation every epoch
        params['original_recipe'] = True
        params['disable_tf32'] = True   # full fp32: TF32 spikes destabilize CG products
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['nb_epochs'] = 100

    elif argv == '57':
        print("FOA + multi ACCDOA + cgseld, TAU-NIGENS 2021, train [1,2,3,4], FAST config (compile + batch 96)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21_cgseld'
        # published recipe except batch size (96 vs 12) + torch.compile;
        # benchmarked best-throughput A100 config (46 samp/s vs 19)
        params['original_recipe'] = True
        params['disable_tf32'] = True
        params['compile'] = True
        params['batch_size'] = 96
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['nb_epochs'] = 100

    elif argv == '58':
        print("FOA + multi ACCDOA + cgseld, STARSS23 fold [3], FAST config (compile + batch 96), 200 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_cgseld'
        # published recipe except batch size (96 vs 12) + torch.compile
        # (task 57 note); features re-extracted with the capped bin scale
        params['original_recipe'] = True
        params['disable_tf32'] = True
        params['compile'] = True
        params['batch_size'] = 96
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        params['extract_folds'] = [3, 4]   # only the folds this run touches

    elif argv == '59':
        print("FOA + multi ACCDOA + cgseld, STARSS23 fold [3], compile + batch 12, 200 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_cgseld'
        # published batch size: at batch 12 a ~0.8% bad-chunk rate skips
        # ~8% of steps (vs >50% at batch 96); val/test run eager (no
        # dynamic-shape OOM). Features: capped bin scale, folds 3-4.
        params['original_recipe'] = True
        params['disable_tf32'] = True
        params['compile'] = True
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        params['extract_folds'] = [3, 4]

    elif argv == '60':
        print("cgseld TAU2021 fold [1], reference StandardScaler features (normalization ablation)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['original_recipe'] = True
        params['disable_tf32'] = True
        params['compile'] = True
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['train_splits'] = [1]
        params['nb_epochs'] = 100
        params['cgseld_std_scaler'] = True
        params['extract_folds'] = [1, 5, 6]
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21_cgseld_std'

    elif argv == '61':
        print("cgseld TAU2021 fold [1], per-bin scale features (normalization ablation control, parity code)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['original_recipe'] = True
        params['disable_tf32'] = True
        params['compile'] = True
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['train_splits'] = [1]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21_cgseld'

    elif argv == '62':
        print("NON-EQUIVARIANT TWIN of equiseld, TAU-NIGENS 2021, train [1,2,3,4] (mirrors task 40)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '63':
        print("NON-EQUIVARIANT TWIN of equiseld, TAU-NIGENS 2021, train [1] only (mirrors task 42)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '64':
        print("NON-EQUIVARIANT TWIN of equiseld, STARSS23 fold [3], const-lr recipe (mirrors task 28)\n")
        params.update(quick_test=False, dataset='foa', multi_accdoa=True, model='equiseld')
        params['equiseld_equivariant'] = False
        params['train_splits'] = [3]; params['nb_epochs'] = 200; params['lr_floor'] = 1.0
        params['eval_every_n_epochs'] = 10; params['equiseld_batch'] = 128
        params['equiseld_warmup'] = 500

    elif argv == '65':
        print("NON-EQUIVARIANT TWIN of equiseld, FRONTAL-restricted train, full-sphere test (mirrors task 14)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_frontal'
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10

    elif argv == '66':
        print("NON-EQUIVARIANT TWIN of equiseld, TAU-NIGENS 2021, train [1, 2] (mirrors task 43)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1, 2]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '67':
        print("NON-EQUIVARIANT TWIN of equiseld, TAU-NIGENS 2021, train [1, 2, 3] (mirrors task 44)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1, 2, 3]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '68':
        print("cgseld STARSS23 fold [3], compile + batch 12, VALIDATE EVERY EPOCH, 60 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_cgseld'
        # published recipe; per-epoch validation (no eval_every_n_epochs) so
        # the fp32-inference question is answered at epoch 0 and the
        # overfitting turn (epoch ~6-21 on TAU) is captured exactly.
        # 60 epochs: fold-4 validation costs ~10 min/pass, and the best
        # checkpoint on this model has always come early.
        params['original_recipe'] = True
        params['disable_tf32'] = True
        params['compile'] = True
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['train_splits'] = [3]
        params['nb_epochs'] = 60
        params['extract_folds'] = [3, 4]

    elif argv == '69':
        print("cgseld STARSS23 fold [3], REFERENCE StandardScaler features, val every epoch, 60 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        # mirrors task 68 exactly except the input normalization: per-column
        # (bin x channel x Re/Im) StandardScaler with mean subtraction, fit
        # on the training fold, as in the reference implementation. The mean
        # shift breaks exact rotation equivariance; it also bounds the
        # dynamic range per channel, which may cure the fp32 overflow.
        params['cgseld_std_scaler'] = True
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_cgseld_std'
        params['original_recipe'] = True
        params['disable_tf32'] = True
        params['compile'] = True
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['train_splits'] = [3]
        params['nb_epochs'] = 60
        params['extract_folds'] = [3, 4]

    elif argv == '70':
        print("NON-EQUIVARIANT TWIN + waveform O(3) aug, TAU-NIGENS 2021, train [1,2,3,4] (mirrors task 62)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        params['wave_aug'] = True
        params['wave_num_workers'] = 12

    elif argv == '71':
        print("NON-EQUIVARIANT TWIN + waveform O(3) aug, STARSS23 fold [3] (mirrors task 64)\n")
        params.update(quick_test=False, dataset='foa', multi_accdoa=True,
                      model='equiseld')
        params['equiseld_equivariant'] = False
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['lr_floor'] = 1.0
        params['eval_every_n_epochs'] = 10
        params['equiseld_batch'] = 128
        params['equiseld_warmup'] = 500
        params['wave_aug'] = True
        params['wave_num_workers'] = 12

    elif argv == '72':
        print("NON-EQUIVARIANT TWIN + waveform O(3) aug, TAU-NIGENS 2021, train [1] (mirrors task 63)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        params['wave_aug'] = True
        params['wave_num_workers'] = 12

    elif argv == '73':
        print("NON-EQUIVARIANT TWIN + waveform O(3) aug, TAU-NIGENS 2021, train [1, 2] (mirrors task 66)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1, 2]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        params['wave_aug'] = True
        params['wave_num_workers'] = 12

    elif argv == '74':
        print("NON-EQUIVARIANT TWIN + waveform O(3) aug, TAU-NIGENS 2021, train [1, 2, 3] (mirrors task 67)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1, 2, 3]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        params['wave_aug'] = True
        params['wave_num_workers'] = 12

    elif argv == '75':
        print("PARAM-MATCHED NON-EQUIVARIANT TWIN (d_v=12), "
              "TAU-NIGENS 2021, train [1,2,3,4] (mirrors task 62)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        # d_v 12 (not 32) gives 2,249,757 params, within 0.13% of
        # equiseld's 2,252,716. The vector path then carries 12
        # channels of raw components (36 injection dims) instead of
        # 32 channels of equivariant vectors (32 injection dims),
        # so the scalar/attention path sees a comparable width too.
        params['equiseld_d_v'] = 12

    elif argv == '76':
        print("PARAM-MATCHED NON-EQUIVARIANT TWIN (d_v=12), "
              "TAU-NIGENS 2021, train [1] only (mirrors task 63)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        # d_v 12 (not 32): 2,249,757 params, within 0.13% of
        # equiseld's 2,252,716 — the vector path carries 12
        # channels of raw components (36 injection dims) instead of
        # 32 channels of equivariant vectors (32 injection dims),
        # so the scalar/attention path sees a comparable width too.
        params['equiseld_d_v'] = 12
    elif argv == '77':
        print("PARAM-MATCHED NON-EQUIVARIANT TWIN (d_v=12) + waveform "
              "O(3) aug, TAU-NIGENS 2021, train [1] (mirrors task 72)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        params['equiseld_d_v'] = 12      # param-matched: 2,249,757
        params['wave_aug'] = True    # fresh Haar O(3) per chunk
        params['wave_num_workers'] = 12
    elif argv == '78':
        print("PARAM-MATCHED NON-EQUIVARIANT TWIN (d_v=12) + waveform "
              "O(3) aug, TAU-NIGENS 2021, train [1, 2] "
              "(mirrors task 73)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        params['equiseld_d_v'] = 12            # param-matched
        params['wave_aug'] = True
        params['wave_num_workers'] = 12
        params['train_splits'] = [1, 2]

    elif argv == '79':
        print("PARAM-MATCHED NON-EQUIVARIANT TWIN (d_v=12) + waveform "
              "O(3) aug, TAU-NIGENS 2021, train [1, 2, 3] "
              "(mirrors task 74)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        params['equiseld_d_v'] = 12            # param-matched
        params['wave_aug'] = True
        params['wave_num_workers'] = 12
        params['train_splits'] = [1, 2, 3]

    elif argv == '80':
        print("PARAM-MATCHED NON-EQUIVARIANT TWIN (d_v=12) + waveform "
              "O(3) aug, TAU-NIGENS 2021, train [1, 2, 3, 4] "
              "(mirrors task 70)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        params['equiseld_d_v'] = 12            # param-matched
        params['wave_aug'] = True
        params['wave_num_workers'] = 12
    elif argv == '81':
        print("PARAM-MATCHED NON-EQUIVARIANT TWIN (d_v=12), "
              "TAU-NIGENS 2021, train [1, 2] (mirrors task 66)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1, 2]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        params['equiseld_d_v'] = 12            # param-matched

    elif argv == '82':
        print("PARAM-MATCHED NON-EQUIVARIANT TWIN (d_v=12), "
              "TAU-NIGENS 2021, train [1, 2, 3] (mirrors task 67)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_equivariant'] = False
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['train_splits'] = [1, 2, 3]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'
        params['equiseld_d_v'] = 12            # param-matched

    elif argv == '83':
        print("PARAM-MATCHED NON-EQUIVARIANT TWIN (d_v=12), "
              "STARSS23 fold [3] (mirrors task 64)\n")
        params.update(quick_test=False, dataset='foa',
                      multi_accdoa=True, model='equiseld')
        params['equiseld_equivariant'] = False
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['lr_floor'] = 1.0
        params['eval_every_n_epochs'] = 10
        params['equiseld_batch'] = 128
        params['equiseld_warmup'] = 500
        params['equiseld_d_v'] = 12            # param-matched

    elif argv == '84':
        print("PARAM-MATCHED NON-EQUIVARIANT TWIN (d_v=12) + waveform O(3) aug, "
              "STARSS23 fold [3] (mirrors task 71)\n")
        params.update(quick_test=False, dataset='foa',
                      multi_accdoa=True, model='equiseld')
        params['equiseld_equivariant'] = False
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['lr_floor'] = 1.0
        params['eval_every_n_epochs'] = 10
        params['equiseld_batch'] = 128
        params['equiseld_warmup'] = 500
        params['equiseld_d_v'] = 12            # param-matched
        params['wave_aug'] = True
        params['wave_num_workers'] = 12
    elif argv == '85':
        print("cgseld STARSS23 fold [3], StandardScaler, 5-epoch smoke run\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        # mirrors task 68 exactly except the input normalization: per-column
        # (bin x channel x Re/Im) StandardScaler with mean subtraction, fit
        # on the training fold, as in the reference implementation. The mean
        # shift breaks exact rotation equivariance; it also bounds the
        # dynamic range per channel, which may cure the fp32 overflow.
        params['cgseld_std_scaler'] = True
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_cgseld_std'
        params['original_recipe'] = True
        params['disable_tf32'] = True
        params['compile'] = True
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['train_splits'] = [3]
        params['extract_folds'] = [3, 4]
        params['nb_epochs'] = 5     # smoke run: time the
        # compiled validation pass

    elif argv == '86':
        print("cgseld TAU2021 fold [1], StandardScaler, 5-epoch smoke run\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['original_recipe'] = True
        params['disable_tf32'] = True
        params['compile'] = True
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['train_splits'] = [1]
        params['cgseld_std_scaler'] = True
        params['extract_folds'] = [1, 5, 6]
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21_cgseld_std'
        params['nb_epochs'] = 5     # smoke run: time the
        # compiled validation pass
    elif argv == '87':
        print("cgseld TAU2021, full train folds [1-4], StandardScaler, "
              "5-epoch smoke run\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['original_recipe'] = True
        params['disable_tf32'] = True
        params['compile'] = True
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['cgseld_std_scaler'] = True
        # full protocol: train 1-4, val 5, test 6 (no train_splits
        # override), so every fold needs features
        params['extract_folds'] = [1, 2, 3, 4, 5, 6]
        params['nb_epochs'] = 5
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21_cgseld_std_full'
    elif argv == '88':
        print("cgseld STARSS23 fold [3], StandardScaler, TF32 ENABLED, "
              "5-epoch smoke run\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        # mirrors task 68 exactly except the input normalization: per-column
        # (bin x channel x Re/Im) StandardScaler with mean subtraction, fit
        # on the training fold, as in the reference implementation. The mean
        # shift breaks exact rotation equivariance; it also bounds the
        # dynamic range per channel, which may cure the fp32 overflow.
        params['cgseld_std_scaler'] = True
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_cgseld_std'
        params['original_recipe'] = True
        params['compile'] = True
        params['batch_size'] = 12
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['train_splits'] = [3]
        params['extract_folds'] = [3, 4]
        params['nb_epochs'] = 5     # smoke run: time the
        # compiled validation pass
        params['enable_tf32'] = True   # tensor-core matmuls;
        # watch the non-finite counter, this is why it was off
    elif argv == '89':
        print("cgseld TAU2021 full protocol, StandardScaler, TF32, "
              "100 epochs, validation every epoch\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['original_recipe'] = True
        params['compile'] = True
        params['enable_tf32'] = True   # tensor-core matmuls
        params['batch_size'] = 48
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['cgseld_std_scaler'] = True
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        # no train_splits override: DCASE2021 protocol trains on
        # folds 1-4, validates on 5, tests on 6 -- so all folds need
        # features
        params['extract_folds'] = [1, 2, 3, 4, 5, 6]
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21_cgseld_std_full'

    elif argv == '90':
        print("cgseld STARSS23 fold [3], StandardScaler, TF32, "
              "200 epochs, validation every 10\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['original_recipe'] = True
        params['compile'] = True
        params['enable_tf32'] = True   # tensor-core matmuls
        params['batch_size'] = 48
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['cgseld_std_scaler'] = True
        params['train_splits'] = [3]
        params['extract_folds'] = [3, 4]
        params['nb_epochs'] = 200
        params['eval_every_n_epochs'] = 10
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_cgseld_std'
    elif argv == '91':
        print("cgseld TAU2021, ORIGINAL single-DOA head and objective, "
              "StandardScaler, TF32, 100 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False   # K=1, one DOA per class
        params['cgseld_original_head'] = True   # BCE + angular loss
        params['model'] = 'cgseld'
        params['nb_mel_bins'] = 1024
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['original_recipe'] = True
        params['compile'] = True
        params['enable_tf32'] = True
        params['batch_size'] = 48
        params['lr'] = 1e-3
        params['weight_decay'] = 1e-5
        params['label_sequence_length'] = 26
        params['cgseld_std_scaler'] = True
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21_cgseld_std_full'
    elif argv == '92':
        print("SO(3)-RESTRICTED EquiSELD (axial v_a x v_r channel), "
              "TAU-NIGENS 2021 (mirrors task 40)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_so3_only'] = True
        params['dataset_dir'] = '/projects/0/prjs1261/seld/TAU2021'
        params['nb_epochs'] = 100
        params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_tau21'

    elif argv == '93':
        print("SO(3)-RESTRICTED EquiSELD (axial v_a x v_r channel), "
              "STARSS23 fold [3] (mirrors task 28)\n")
        params.update(quick_test=False, dataset='foa',
                      multi_accdoa=True, model='equiseld')
        params['equiseld_so3_only'] = True
        params['train_splits'] = [3]
        params['nb_epochs'] = 200
        params['lr_floor'] = 1.0
        params['eval_every_n_epochs'] = 10
        params['equiseld_batch'] = 128
        params['equiseld_warmup'] = 500
    elif argv == '94':
        print("LARGE SO(3)-RESTRICTED EquiSELD (d_s 384 / d_v 96 / 8 heads, "
              "20.0M params), full STARSS23 [1,2,3], 250 epochs\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'equiseld'
        params['equiseld_so3_only'] = True
        # capacity: 20,035,885 params (task 8 is 2,252,716)
        params['equiseld_d_s'] = 384
        params['equiseld_d_v'] = 96
        params['equiseld_heads'] = 8
        # block counts, batch, lr, warmup, wd, dropout: task 8 defaults
        params['nb_epochs'] = 250
        # val_splits == test_splits == [4] on STARSS23, so every validation
        # is another best-of-trajectory selection on the test fold. Task 36
        # (2.25M, full data) selected over 20 checkpoints; 25 here keeps the
        # scaling delta from being inflated by a richer selection budget.
        params['eval_every_n_epochs'] = 10
    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    if params.get('model', 'seldnet') == 'equiseld':
        # Separate feature/label root: equiseld features (FoaFeatures front
        # end, 11 ch, unnormalized) must never mix with baseline features.
        if params['feat_label_dir'] == '/projects/0/prjs1261/seld/STARSS2023/seld_feat_label':
            # default -> equiseld's own 24k feature root on scratch
            params['feat_label_dir'] = '/scratch-shared/gyuksel2/seld/seld_feat_label_24k_equiseld'
        else:
            # task chose a custom root (e.g. frontal-restricted): suffix it
            params['feat_label_dir'] = params['feat_label_dir'].rstrip('/') + '_equiseld'
        # Training recipe validated by the equiseld overfit test (see the
        # "Training recipe" section of equiseld.py's module docstring):
        # AdamW 3e-4, wd 0.05 (with exclusions), warmup+cosine, grad-clip 1.0,
        # annealed SED-BCE warm-up against detection starvation under ADPIT.
        params['batch_size'] = params.pop('equiseld_batch', 32)
        params['lr'] = params.pop('equiseld_lr', 3e-4)
        params['weight_decay'] = 0.05
        params['warmup_steps'] = params.pop('equiseld_warmup', 2000)
        params['grad_clip'] = 1.0
        # Architecture overrides (defaults = equiseld.SeldConfig; see
        # equiseld_dcase.make_seld_config for all recognized keys)
        params['equiseld_f_vec_max'] = 9000.0

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14 
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13


    # node-local staging: the sbatch 'stage' option copies the task's
    # feat/label dirs to $TMPDIR (node NVMe) and sets this override
    import os
    if os.environ.get('SELD_FEAT_DIR_OVERRIDE'):
        params['feat_label_dir'] = os.environ['SELD_FEAT_DIR_OVERRIDE']

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
