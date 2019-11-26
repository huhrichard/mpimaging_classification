from utils.common_library import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd

def compare_model(trainers, save_path, output_label='', output_idx=0, multi_label_classify=False, metrics=None):

    n = len(trainers)
    colors = plt.cm.jet(np.linspace(0, 1, n*3))
    maxepochs = 0
    plt.clf()
    print(trainers[0][0])
    if metrics is None:
        metrics = list(trainers[0][0].performance_stat["train"][0].keys())
    print(metrics)
    epochs = trainers[0][0].total_epochs
    states = ["train", "val", "test"]
    linestyle_dict = {"train": "--",
                      "val": "--",
                      "test": "--"}
    mark_style_dict = {"train": ".",
                      "val": "^",
                      "test": "D"}
    c_style = {"train": 0,
               "val": 1,
               "test": 2}
    """
    for loop: metric->trainer
                             ->concat each fold data
                             ->plot with each trainer
                    ->plot all trainer
    """
    for metric in metrics:
        fig_all_trainers = plt.figure()
        ax_all_trainers = fig_all_trainers.add_subplot(1,1,1)
        fig_all_trainers_test_only = plt.figure()
        ax_all_trainers_test_only = fig_all_trainers_test_only.add_subplot(1, 1, 1)
        fig_all_trainers_train_only = plt.figure()
        ax_all_trainers_train_only = fig_all_trainers_train_only.add_subplot(1, 1, 1)
        fig_all_trainers_val_only = plt.figure()
        ax_all_trainers_val_only = fig_all_trainers_val_only.add_subplot(1, 1, 1)
        for idx, specific_trainer in enumerate(trainers):
            # plot each fold result 1st
            fig_all_folds = plt.figure()
            ax_all_folds = fig_all_folds.add_subplot(1,1,1)
            fig_all_folds_train_only = plt.figure()
            ax_all_folds_train_only = fig_all_folds_train_only.add_subplot(1, 1, 1)
            fig_all_folds_val_only = plt.figure()
            ax_all_folds_val_only = fig_all_folds_val_only.add_subplot(1, 1, 1)
            fig_all_folds_test_only = plt.figure()
            ax_all_folds_test_only = fig_all_folds_test_only.add_subplot(1,1,1)

            metric_dicts = {"train":[],
                            "val":[],
                            "test":[]
                            }
            for nth_fold in range(len(specific_trainer)):
                nth_fold_trainer = specific_trainer[nth_fold]

                for state in states:
                    if multi_label_classify:
                        metric_scores = [nth_fold_trainer.performance_stat[state][e][metric][output_idx] for e in range(epochs)]
                    else:
                        metric_scores = [nth_fold_trainer.performance_stat[state][e][metric] for e in range(epochs)]

                    metric_dicts[state].append(metric_scores)

            for state in states:
                metric_dicts[state] = np.array(metric_dicts[state])
                plot_paras = {"x": range(epochs),
                              "y": np.mean(metric_dicts[state], axis=0),
                              "yerr": np.std(metric_dicts[state], axis=0),
                              "label": "{}({})".format(state, specific_trainer[0].model_name),
                              "c": colors[idx*3+c_style[state]],
                              "linestyle": linestyle_dict[state],
                              "alpha": 0.4,
                              "marker":mark_style_dict[state]}
                if state == "test":
                    ax_all_folds_test_only.errorbar(**plot_paras)
                    ax_all_trainers_test_only.errorbar(**plot_paras)
                elif state == "train":
                    ax_all_folds_train_only.errorbar(**plot_paras)
                    ax_all_trainers_train_only.errorbar(**plot_paras)
                else:
                    ax_all_folds_val_only.errorbar(**plot_paras)
                    ax_all_trainers_val_only.errorbar(**plot_paras)


                ax_all_folds.errorbar(**plot_paras)
                ax_all_trainers.errorbar(**plot_paras)

            base_name = "{}_{}".format(metric, specific_trainer[0].model_name)
            ax_all_folds.set_title("{} of {}".format(metric, output_label))
            ax_all_folds.legend()
            fig_all_folds.savefig(save_path + base_name + ".png")
            fig_all_folds.clf()

            ax_all_folds_test_only.set_title("{} of {}".format(metric, output_label))
            ax_all_folds_test_only.legend()
            fig_all_folds_test_only.savefig(save_path + base_name + "_test.png")
            fig_all_folds_test_only.clf()

            ax_all_folds_train_only.set_title("{} of {}".format(metric, output_label))
            ax_all_folds_train_only.legend()
            fig_all_folds_train_only.savefig(
                save_path + base_name + "_train.png")
            fig_all_folds_train_only.clf()

            ax_all_folds_val_only.set_title("{} of {}".format(metric, output_label))
            ax_all_folds_val_only.legend()
            fig_all_folds_val_only.savefig(
                save_path + base_name + "_val.png")
            fig_all_folds_val_only.clf()

        base_name = "{}".format(metric)

        ax_all_trainers.set_title("{} of {}".format(metric, output_label))
        ax_all_trainers.legend()
        fig_all_trainers.savefig(save_path + base_name + ".png")
        fig_all_trainers.clf()

        ax_all_trainers_test_only.set_title("{} of {}".format(metric, output_label))
        ax_all_trainers_test_only.legend()
        fig_all_trainers_test_only.savefig(save_path + base_name + "_test.png")
        fig_all_trainers_test_only.clf()

        ax_all_trainers_train_only.set_title("{} of {}".format(metric, output_label))
        ax_all_trainers_train_only.legend()
        fig_all_trainers_train_only.savefig(save_path + base_name + "_train.png")
        fig_all_trainers_train_only.clf()

        ax_all_trainers_val_only.set_title("{} of {}".format(metric, output_label))
        ax_all_trainers_val_only.legend()
        fig_all_trainers_val_only.savefig(save_path + base_name + "_val.png")
        fig_all_trainers_val_only.clf()


def compare_model_cv(trainers, save_path, out_csv='',
                     output_label='', output_idx=0,
                     multi_label_classify=False, metrics=None,
                     plot_states_list=None,
                     ):

    n = len(trainers)

    maxepochs = 0
    plt.clf()
    print(trainers[0])
    if metrics is None:
        metrics = list(trainers[0].performance_stat["train"][0].keys())
    print(metrics)
    epochs = trainers[0].total_epochs
    # states = ["train", "val"]
    states = ["val"]
    if plot_states_list is None:
        plot_states_list = ["_val"]
        # plot_states_list = ["_train", "_val", ""]

    colors = plt.cm.jet(np.linspace(0, 1, n * len(states)))

    linestyle_dict = {"train": "--",
                      "val": "--",
                      "test": "--"}
    mark_style_dict = {"train": ".",
                      "val": "^",
                      "test": "D"}
    c_style = {"train": 1,
               "val": 0,
               "test": 2}
    """
    for loop: metric->trainer
                             ->plot with each trainer
                    ->plot all trainer
    """

    for metric in metrics:
        fig_all_trainer_list = [plt.figure() for plot in plot_states_list]
        ax_all_trainer_list = [fig.add_subplot(1,1,1) for fig in fig_all_trainer_list]
        for trainer_idx, specific_trainer in enumerate(trainers):
            # plot each fold result 1st
            fig_all_fold_list = [plt.figure() for plot in plot_states_list]
            ax_all_fold_list = [fig.add_subplot(1, 1, 1) for fig in fig_all_fold_list]
            for plot_idx, plot in enumerate(plot_states_list):
                for state in states:
                    if state in plot or plot == "":

                        plot_paras = {
                                        # "x": range(epochs),
                                      # "y": np.mean(metric_dicts[state], axis=0),
                                      # "yerr": np.std(metric_dicts[state], axis=0),
                                      "label": "{}({})".format(state, specific_trainer.model_name),
                                      "c": colors[trainer_idx * len(states) + c_style[state]],
                                      "linestyle": linestyle_dict[state],
                                      "alpha": 0.4,
                                      "marker": mark_style_dict[state]}

                        if state == "train":
                            # print(specific_trainer.performance_stat[metric][state])
                            if multi_label_classify:

                                plot_paras["y"] = np.mean(specific_trainer.performance_stat[metric][state][:, :, output_idx], axis=0)
                                plot_paras["yerr"] = np.std(specific_trainer.performance_stat[metric][state][:, :, output_idx], axis=0)
                            else:
                                plot_paras["y"] = np.mean(specific_trainer.performance_stat[metric][state], axis=0)
                                plot_paras["yerr"] = np.std(specific_trainer.performance_stat[metric][state], axis=0)
                            plot_paras["x"] = range(epochs)
                            ax_all_fold_list[plot_idx].errorbar(**plot_paras)
                            ax_all_trainer_list[plot_idx].errorbar(**plot_paras)
                        else:
                            # print(state, metric, ': ', specific_trainer.performance_stat[metric][state])
                            if multi_label_classify:
                                y = specific_trainer.performance_stat[metric][state][:, output_idx]
                            else:
                                y = specific_trainer.performance_stat[metric][state]
                            x = range(epochs)
                            ax_all_fold_list[plot_idx].plot(x, y, **plot_paras)
                            ax_all_trainer_list[plot_idx].plot(x, y, **plot_paras)
                base_name = "{}_{}".format(metric, specific_trainer.model_name)
                ax_all_fold_list[plot_idx].legend()
                ax_all_fold_list[plot_idx].set_title("{} of {}".format(metric, output_label))
                fig_all_fold_list[plot_idx].savefig(save_path + base_name + plot + ".png")

        for plot_idx, plot in enumerate(plot_states_list):
            base_name = "{}".format(metric)
            ax_all_trainer_list[plot_idx].legend()
            ax_all_trainer_list[plot_idx].set_title("{} of {}".format(metric, output_label))
            fig_all_trainer_list[plot_idx].savefig(save_path + base_name + plot + ".png")


def write_prediction_on_df(trainers, df, state, patient_dataset, out_label_name, out_label_idx,
                            epoch_as_final = -1,
                            ):
    for trainer in trainers:
        col_pred_name = "{}_{}_{}_prediction".format(out_label_name, state, trainer.model_name)
        # this is patient idx fo patient dataset
        df[col_pred_name] = 0
        idx_list = torch.Tensor([trainer.idx_list[nth_fold][state][epoch_as_final] for nth_fold in range(trainer.n_fold)])
        pred = np.concatenate([trainer.prediction_list[nth_fold][state][epoch_as_final] for nth_fold in range(trainer.n_fold)], axis=0)
        for idx_for_trainer, idx_for_dataset in enumerate(idx_list):
            img_path_list = patient_dataset.patient_img_list[idx_for_dataset]
            for img_idx, img_path in enumerate(img_path_list):
                # img_trimmed_path = img_path.split('/')[-1].split('.')[0]
                p = pred[img_idx+idx_for_trainer*len(img_path_list)][state][epoch_as_final][out_label_idx]
                df.loc[df['MPM image file per TMA core '] in img_path][col_pred_name] = p

    return df
        # df[col_pred_name] = trainer.prediction_list[][state][epoch_as_final]

def write_scores_on_df(trainers, df, metrics, state, out_label='', out_idx=None, epoch_as_final = -1,):

    for trainer in trainers:
        for metric in metrics:
            if out_idx is None:
                col_metric_name = "{}_{}_{}".format(metric, state, trainer.model_name)
                df[col_metric_name] = trainer.performance_stat[metric][state][epoch_as_final]
            else:
                col_metric_name = "{}_{}_{}_{}".format(out_label, metric, state, trainer.model_name)
                df[col_metric_name] = trainer.performance_stat[metric][state][epoch_as_final, out_idx]

    return df
