parameters_grid = {
                   # "epochs": [args.epochs],
                   # "num_classes": [num_classes],
                   "multi_label": [True],
                   # "n_fold": [n_fold],
                   # "performance_metrics_list": [metric_list],
                   # "device": [device],
                   # "p_model": ["resnext101_32x8d"],
                   "p_model": ["resnet18", "resnet50"],
                   # "p_model": ["wide_resnet101_2"],
                   "p_weight": [True],
                   "feat_ext": [False],
                   "lr": [1e-7, 1e-5],
                   "wd": [1e-3],
                   "input_dim": [(224, 224)],
                   "out_list": [False],
                   "loss": ["BCE", "FL"],
                   "train_data_normal": [True],
                   "n_batch": [1, 8, 32],
                   }