#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import seldnet_model
import parameters
import time
from time import gmtime, strftime
import torch
import torch.nn as nn
import torch.optim as optim
plot.switch_backend('agg')
from IPython import embed
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
import seldnet_model
import equiseld_dcase

def get_accdoa_labels(accdoa_in, nb_classes, thresh=0.5):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > thresh

    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes, thresh=0.5):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > thresh
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3*nb_classes:4*nb_classes], accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes]
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > thresh
    doa1 = accdoa_in[:, :, 3*nb_classes: 6*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6*nb_classes:7*nb_classes], accdoa_in[:, :, 7*nb_classes:8*nb_classes], accdoa_in[:, :, 8*nb_classes:]
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > thresh
    doa2 = accdoa_in[:, :, 6*nb_classes:]

    return sed0, doa0, sed1, doa1, sed2, doa2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    # Number of frames for a 60 second audio with 100ms hop length = 600 frames
    # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with zero padding in the remaining frames
    test_filelist = data_generator.get_filelist()

    nb_test_batches, test_loss = 0, 0.
    nb_classes = params['unique_classes']
    mag_min = np.full(nb_classes, np.inf, dtype=np.float32)
    mag_max = np.full(nb_classes, -np.inf, dtype=np.float32)
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for data, target in data_generator.generate():
            # load one batch of data
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

            # process the batch of data based on chosen mode.
            # per_file batches hold a whole recording (up to ~115 sequences),
            # far larger than the training batch; chunk the forward pass so
            # inference memory matches training. Exact in eval mode: norm
            # layers use stored statistics, and no op mixes samples.
            # fixed-size chunks so a compiled model sees one static
            # shape; the last chunk is zero-padded and trimmed back
            chunk = params.get('infer_chunk') or params['batch_size']
            n = data.shape[0]
            if n == chunk:
                output = model(data)
            else:
                outs = []
                for i in range(0, n, chunk):
                    sub = data[i:i + chunk]
                    pad = chunk - sub.shape[0]
                    if pad > 0:
                        sub = torch.cat(
                            [sub, sub.new_zeros((pad,) + tuple(sub.shape[1:]))],
                            dim=0)
                    out = model(sub)
                    outs.append(out[:chunk - pad] if pad > 0 else out)
                output = torch.cat(outs, dim=0)
            loss = criterion(output, target)

            # --- track per-class ACCDOA magnitude range over predictions ---
            out_np = output.detach().cpu().numpy()
            if params['multi_accdoa'] is True:
                mags = []
                for tr in range(3):
                    base = tr * 3 * nb_classes
                    x_t = out_np[..., base               : base +   nb_classes]
                    y_t = out_np[..., base +   nb_classes: base + 2*nb_classes]
                    z_t = out_np[..., base + 2*nb_classes: base + 3*nb_classes]
                    mags.append(np.sqrt(x_t**2 + y_t**2 + z_t**2))  # [B, T, C]
                # stack tracks -> [B, T, num_tracks, C] then reduce all but class axis
                mag = np.stack(mags, axis=-2)
            else:
                x_t = out_np[..., :nb_classes]
                y_t = out_np[..., nb_classes:2*nb_classes]
                z_t = out_np[..., 2*nb_classes:3*nb_classes]
                mag = np.sqrt(x_t**2 + y_t**2 + z_t**2)  # [B, T, C]
            mag_min = np.minimum(mag_min, mag.reshape(-1, nb_classes).min(axis=0))
            mag_max = np.maximum(mag_max, mag.reshape(-1, nb_classes).max(axis=0))
            # ---------------------------------------------------------------

            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(out_np, params['unique_classes'], params.get('sed_threshold', 0.5))
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(out_np, params['unique_classes'], params.get('sed_threshold', 0.5))
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            if params['multi_accdoa'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']]])
                            if sed_pred1[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']]])
                            if sed_pred2[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+params['unique_classes']], doa_pred[frame_cnt][class_cnt+2*params['unique_classes']]]) 
            data_generator.write_output_format_file(output_file, output_dict)

            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break


        test_loss /= nb_test_batches

    return test_loss, mag_min, mag_max


def train_epoch(data_generator, optimizer, model, criterion, params, device,
                scheduler=None):
    nb_train_batches, train_loss = 0, 0.
    nb_finite, nb_skipped = 0, 0
    model.train()
    for data, target in data_generator.generate():
        # load one batch of data
        data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
        optimizer.zero_grad()

        # process the batch of data based on chosen mode
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        # max_norm=inf leaves gradients untouched but still returns the total
        # norm, so the non-finite skip guard works with clipping disabled
        gnorm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), params.get('grad_clip') or float('inf'))
        if not torch.isfinite(gnorm):
            # skip the update instead of writing NaN into the weights
            optimizer.zero_grad(set_to_none=True)
            nb_skipped += 1
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # a non-finite batch loss would poison the epoch average; its step
        # was already skipped above (non-finite loss => non-finite grads)
        if torch.isfinite(loss):
            train_loss += loss.item()
            nb_finite += 1
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break

    if nb_skipped:
        print('  WARNING: {}/{} non-finite batches, steps skipped'.format(
            nb_skipped, nb_train_batches), flush=True)
    train_loss /= max(1, nb_finite)

    return train_loss


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    print(argv, flush=True)
    if len(argv) != 3:
        print('\n\n', flush=True)
        print('-------------------------------------------------------------------------------------------------------', flush=True)
        print('The code expected two optional inputs', flush=True)
        print('\t>> python seld.py <task-id> <job-id>', flush=True)
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py', flush=True)
        print('Using default inputs for now', flush=True)
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.', flush=True)
        print('-------------------------------------------------------------------------------------------------------', flush=True)
        print('\n\n', flush=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(False)  # enable only when debugging NaNs (large slowdown)
    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    if use_cuda and params.get('disable_tf32'):
        # full-fp32 matmuls: the TF32 default on A100 causes gradient
        # spikes in the CG-product network (pre-TF32-era published recipe)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print('TF32 disabled (full float32 matmuls)', flush=True)
    elif use_cuda and params.get('enable_tf32'):
        # opt back in to tensor-core matmuls: faster, but the reason
        # disable_tf32 exists is that the CG products spiked under it —
        # watch the non-finite batch counter when using this
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print('TF32 enabled (matmul precision "high")', flush=True)

    job_id = 1 if len(argv) < 3 else argv[-1]

    # Training setup
    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        if '2020' in params['dataset_dir']:
            test_splits = [1]
            val_splits = [2]
            train_splits = [[3, 4, 5, 6]]

        elif '2021' in params['dataset_dir']:
            test_splits = [6]
            val_splits = [5]
            train_splits = [[1, 2, 3, 4]]

        elif '2022' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]] 
        elif '2023' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]] 


        else:
            print('ERROR: Unknown dataset splits', flush=True)
            exit()

        # explicit train-fold override (data-fraction experiments)
        if params.get('train_splits'):
            train_splits = [list(params['train_splits'])]
            print('train split override: {}'.format(train_splits), flush=True)
    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------', flush=True)
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split), flush=True)
        print('---------------------------------------------------------------------------------------------------', flush=True)

        # Unique name for the run
        loc_feat = params['dataset']
        if params['dataset'] == 'mic':
            if params['use_salsalite']:
                loc_feat = '{}_salsa'.format(params['dataset'])
            else:
                loc_feat = '{}_gcc'.format(params['dataset'])
        loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_split{}_{}_{}'.format(
            task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat
        )
        model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))
        print("unique_name: {}\n".format(unique_name), flush=True)

        # Load train and validation data
        print('Loading training dataset:', flush=True)
        if params.get('cov_aug', False):
            # continuous O(3) augmentation on rotation-complete covariance
            # features (train only; val/test keep the baseline generator)
            import cov_augmentation
            data_gen_train = cov_augmentation.CovRotDataGenerator(
                params=params, split=train_splits[split_cnt]
            )
        elif params.get('wave_aug', False):
            # on-the-fly waveform-domain O(3) augmentation (train only;
            # val/test below keep the precomputed-feature generator)
            import cls_waveform_generator
            data_gen_train = cls_waveform_generator.WaveformDataGenerator(
                params=params, split=train_splits[split_cnt]
            )
        else:
            data_gen_train = cls_data_generator.DataGenerator(
                params=params, split=train_splits[split_cnt]
            )

        print('Loading validation dataset:', flush=True)
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False, per_file=True
        )

        # Collect i/o data size and load model configuration
        data_in, data_out = data_gen_train.get_data_sizes()
        if params.get('model', 'seldnet') == 'equiseld':
            model = equiseld_dcase.EquiSeldModel(data_in, data_out, params).to(device)
        elif params.get('model', 'seldnet') == 'cgseld':
            import cgseld_dcase
            model = cgseld_dcase.CGSeldModel(data_in, data_out, params).to(device)
        else:
            model = seldnet_model.SeldModel(data_in, data_out, params).to(device)
        if params.get('compile'):
            # keep the eager module reachable via _orig_mod: state_dict keys
            # must stay unprefixed so checkpoints load without compile
            model = torch.compile(model)
            print('torch.compile enabled', flush=True)
        if params['finetune_mode']:
            print('Running in finetuning mode. Initializing the model to the weights - {}'.format(params['pretrained_model_weights']), flush=True)
            getattr(model, '_orig_mod', model).load_state_dict(
                torch.load(params['pretrained_model_weights'], map_location='cpu'))

        print('---------------- SELD-net -------------------', flush=True)
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out), flush=True)
        print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n, rnn_size: {}\n, nb_attention_blocks: {}\n, fnn_size: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'], params['nb_self_attn_layers'],
            params['fnn_size']), flush=True)
        print(model, flush=True)

        # Dump results in DCASE output format for calculating final scores
        dcase_output_val_folder = os.path.join(params['dcase_output_dir'], '{}_{}_val'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
        print('Dumping recording-wise val results in: {}'.format(dcase_output_val_folder), flush=True)

        # Initialize evaluation metric class
        score_obj = ComputeSELDResults(params)

        # start training
        best_val_epoch = -1
        best_ER, best_F, best_LE, best_LR, best_seld_scr = 1., 0., 180., 0., 9999 
        patience_cnt = 0

        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        if params.get('original_recipe'):
            # published recipe of the reimplemented method (Sato et al.:
            # plain Adam, constant lr, uniform weight decay, no warmup)
            optimizer = optim.Adam(model.parameters(), lr=params['lr'],
                                   weight_decay=params.get('weight_decay', 0.0))
            scheduler = None
        elif params.get('model', 'seldnet') in ('equiseld', 'cgseld'):
            # our recipe: AdamW with decay exclusions + warmup/cosine decay
            if params['model'] == 'cgseld':
                import cgseld_dcase
                optimizer = cgseld_dcase.build_optimizer(
                    model, lr=params['lr'],
                    weight_decay=params.get('weight_decay', 0.05))
            else:
                optimizer = equiseld_dcase.build_optimizer(
                    model, lr=params['lr'],
                    weight_decay=params.get('weight_decay', 0.05))
            total_steps = max(1, nb_epoch * data_gen_train.get_total_batches_in_data())
            scheduler = equiseld_dcase.make_scheduler(
                optimizer, params.get('warmup_steps', 2000), total_steps,
                floor=params.get('lr_floor', 0.01))
        else:
            optimizer = optim.Adam(model.parameters(), lr=params['lr'])
            scheduler = None
        if params.get('cgseld_original_head'):
            # Sato et al.'s own objective for their single-DOA head:
            # BCE on activity + masked angular error (not ACCDOA MSE)
            import cgseld_dcase
            criterion = cgseld_dcase.OriginalHeadLoss(
                getattr(model, '_orig_mod', model))
            print('using CGNet original head loss (BCE + angular)',
                  flush=True)
        elif params['multi_accdoa'] is True:
            criterion = seldnet_model.MSELoss_ADPIT()
        else:
            criterion = nn.MSELoss()

        if params.get('val_first'):
            # one validation pass before training: surfaces inference-time
            # OOM / non-finite inference in minutes rather than after a
            # full epoch of training
            print('val_first: validating before training', flush=True)
            vl, vmin, vmax = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)
            print('val_first done: val_loss {:.4f}, |accdoa| in [{:.4f}, {:.4f}]'.format(vl, vmin, vmax), flush=True)

        for epoch_cnt in range(nb_epoch):
            # ---------------------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------------
            start_time = time.time()
            train_loss = train_epoch(data_gen_train, optimizer, model, criterion, params, device,
                                     scheduler=scheduler)
            train_time = time.time() - start_time

            # ---------------------------------------------------------------------
            # VALIDATION (optionally every N epochs; last epoch always)
            # ---------------------------------------------------------------------
            val_every = int(params.get('eval_every_n_epochs', 1))
            if (epoch_cnt + 1) % val_every != 0 and epoch_cnt != nb_epoch - 1:
                print('epoch: {}, time: {:0.2f}, train_loss: {:0.4f} '
                      '(val skipped)'.format(epoch_cnt, train_time, train_loss),
                      flush=True)
                patience_cnt += 1
                if patience_cnt > params['patience']:
                    break
                continue

            start_time = time.time()
            # validate/test with the eager module: per-file batches vary in
            # size, and compiled dynamic-shape graphs recompile and can OOM
            val_loss, val_mag_min, val_mag_max = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)

            # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores
            val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_folder)

            val_time = time.time() - start_time
            
            # Save model if loss is good
            if val_seld_scr <= best_seld_scr:
                best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr
                torch.save(getattr(model, '_orig_mod', model).state_dict(),
                           model_name)

            # Print stats
            print(
                'epoch: {}, time: {:0.2f}/{:0.2f}, '
                # 'train_loss: {:0.2f}, val_loss: {:0.2f}, '
                'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                'ER/F/LE/LR/SELD: {}, '
                'best_val_epoch: {} {}'.format(
                    epoch_cnt, train_time, val_time,
                    train_loss, val_loss,
                    '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr),
                    best_val_epoch, '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_ER, best_F, best_LE, best_LR, best_seld_scr))
            , flush=True)
            print('  val ACCDOA magnitude per class:', flush=True)
            print('  Class\tmin\tmax', flush=True)
            for cls_cnt in range(params['unique_classes']):
                print('  {}\t{:0.4f}\t{:0.4f}'.format(cls_cnt, val_mag_min[cls_cnt], val_mag_max[cls_cnt]), flush=True)

            if params['average'] == 'macro':
                print('  Classwise val results:', flush=True)
                print('  Class\tER\tF\tLE\tLR\tSELD', flush=True)
                for cls_cnt in range(params['unique_classes']):
                    print('  {}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
                        cls_cnt,
                        classwise_val_scr[0][cls_cnt],
                        classwise_val_scr[1][cls_cnt],
                        classwise_val_scr[2][cls_cnt],
                        classwise_val_scr[3][cls_cnt],
                        classwise_val_scr[4][cls_cnt],
                    ), flush=True)

            patience_cnt += 1
            if patience_cnt > params['patience']:
                break

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # ---------------------------------------------------------------------
        print('Load best model weights', flush=True)
        getattr(model, '_orig_mod', model).load_state_dict(
            torch.load(model_name, map_location='cpu'))

        print('Loading unseen test dataset:', flush=True)
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
        )

        # Dump results in DCASE output format for calculating final scores
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder), flush=True)


        test_loss, test_mag_min, test_mag_max = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)
        print('Test ACCDOA magnitude per class:', flush=True)
        print('Class\tmin\tmax', flush=True)
        for cls_cnt in range(params['unique_classes']):
            print('{}\t{:0.4f}\t{:0.4f}'.format(cls_cnt, test_mag_min[cls_cnt], test_mag_max[cls_cnt]), flush=True)

        use_jackknife=True
        print("Getting Test Results", flush=True)
        test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife )
        print('\nTest Loss', flush=True)
        print('SELD score (early stopping metric): {:0.2f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''), flush=True)
        print('SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(test_ER[0]  if use_jackknife else test_ER, '[{:0.2f}, {:0.2f}]'.format(test_ER[1][0], test_ER[1][1]) if use_jackknife else '', 100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''), flush=True)
        print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else '', 100*test_LR[0]  if use_jackknife else 100*test_LR,'[{:0.2f}, {:0.2f}]'.format(100*test_LR[1][0], 100*test_LR[1][1]) if use_jackknife else ''), flush=True)
        if params['average']=='macro':
            print('Classwise results on unseen test data', flush=True)
            print('Class\tER\tF\tLE\tLR\tSELD_score', flush=True)
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                     cls_cnt,
                     classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0], classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0], classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0], classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0], classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0], classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''), flush=True)



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

