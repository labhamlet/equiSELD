# Extracts the features, labels, and normalizes the development and evaluation split features.

import cls_feature_class
import parameters
import sys


def main(argv):
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file.
    # Extracts features and labels relevant for the task-id
    # It is enough to compute the feature and labels once. 

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params)

    if params.get('model', 'seldnet') == 'equiseld':
        # equiseld brings its own SO(3)-equivariant front end; the baseline
        # mel+IV features and their per-dimension normalization are not used
        # (per-component normalization would break exact equivariance).
        import equiseld_dcase
        equiseld_dcase.extract_all_equiseld_features(params)
    elif params.get('model', 'seldnet') == 'cgseld':
        # complex-STFT SH features for the Sato et al. reimplementation
        import cgseld_dcase
        cgseld_dcase.extract_all_cgseld_features(params)
    elif params.get('cov_aug', False):
        # rotation-complete covariance features (O(3) aug for the CRNN)
        import cov_augmentation
        cov_augmentation.extract_all_cov_features(params)
    else:
        # # Extract features and normalize them
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()

    # # Extract labels
    dev_feat_cls.extract_all_labels()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

