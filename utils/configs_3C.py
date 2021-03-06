metric_list = [
                # "f1_by_sample",
               "auc_by_label", "ap_by_label", "fmax_by_label",
               "rmax_by_label", "pmax_by_label",
               "f1_by_label", "balanced_acc_by_label",
               ]
img_metric_list = [
                # "f1_by_sample",
               "auc_by_img", "ap_by_img", "fmax_by_img",
               "rmax_by_img", "pmax_by_img",
               # "f1_by_img", "balanced_acc_by_img",
               ]

patient_metric_list = [
                # "f1_by_sample",
               "auc_by_patient", "ap_by_patient", "fmax_by_patient",
               "rmax_by_patient", "pmax_by_patient",
               # "f1_by_patient", "balanced_acc_by_patient",
               ]
parameters_grid = {
                   # "epochs": [args.epochs],
                   # "num_classes": [num_classes],
                   "multi_label": [True],
                   # "n_fold": [n_fold],
                   # "performance_metrics_list": [metric_list],
                   # "device": [device],
                   # "p_model": ["resnext101_32x8d"],
                   "p_model": ["resnet18"],
                   # "p_model": ["wide_resnet101_2"],
                   "p_weight": [True],
                   "feat_ext": [False],
                   "lr": [1e-7],
                   "wd": [1e-3],
                   "input_dim": [(3, 224, 224)],
                   "out_list": [False],
                   "loss": ["BCE"],
                   "train_data_normal": [True],
                   "n_batch": [1],
                   }