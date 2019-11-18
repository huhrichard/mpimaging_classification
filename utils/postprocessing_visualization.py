from utils.common_library import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def compare_model(trainers, save_path):

    n = len(trainers)
    colors = plt.cm.jet(np.linspace(0, 1, n*3))
    maxepochs = 0
    plt.clf()
    print(trainers[0][0])
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
            ax_all_folds_train_only = fig_all_folds_test_only.add_subplot(1, 1, 1)
            fig_all_folds_val_only = plt.figure()
            ax_all_folds_val_only = fig_all_folds_test_only.add_subplot(1, 1, 1)
            fig_all_folds_test_only = plt.figure()
            ax_all_folds_test_only = fig_all_folds_test_only.add_subplot(1,1,1)

            metric_dicts = {"train":[],
                            "val":[],
                            "test":[]
                            }
            for nth_fold in range(len(specific_trainer)):
                nth_fold_trainer = specific_trainer[nth_fold]
                # fig_all_states = plt.figure()
                # ax_all_states = fig_all_states.add_subplot(1,1,1)

                for state in states:
                    # fig_temp = plt.figure()
                    # ax_temp = fig_temp.add_subplot(1,1,1)
                    # print(nth_fold_trainer)
                    metric_scores = [nth_fold_trainer.performance_stat[state][e][metric] for e in range(epochs)]

                    metric_dicts[state].append(metric_scores)
                    # stacked_fold_metrics.append(metric_scores)

                    # plot_paras = {"label":"{}({})".format(state, nth_fold_trainer.model_name),
                    #               "c":colors[idx*3+c_style[state]],
                    #               "linestyle":linestyle_dict[state],
                    #               "alpha": 0.5,
                    #               "marker":mark_style_dict[state]}
                    # ax_temp.plot(range(epochs), metric_scores,**plot_paras)
                    # ax_all_states.plot(range(epochs), metric_scores, **plot_paras)


                    # ax_temp.set_title(metric)
                    # ax_temp.legend()
                    # ax_temp.set_xlabel("epoch")
                    # ax_temp.set_ylabel(metric)
                    # fig_temp.savefig(save_path+"{}_{}thfold_{}_{}.png".format(metric, nth_fold_trainer.model_name,
                    #                                                        nth_fold, state))
                    # fig_temp.clf()

                # ax_all_states.set_title(metric)
                # ax_all_states.legend()
                # fig_all_states.savefig(save_path+"{}_{}thfold_{}.png".format(metric, nth_fold_trainer.model_name,
                #                                                            nth_fold))
                # fig_all_states.clf()

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

            ax_all_folds.set_title(metric)
            ax_all_folds.legend()
            fig_all_folds.savefig(save_path + "{}_{}.png".format(metric, specific_trainer[0].model_name))
            fig_all_folds.clf()

            ax_all_folds_test_only.set_title(metric)
            ax_all_folds_test_only.legend()
            fig_all_folds_test_only.savefig(save_path + "{}_{}_test.png".format(metric, specific_trainer[0].model_name))
            fig_all_folds_test_only.clf()

            ax_all_folds_train_only.set_title(metric)
            ax_all_folds_train_only.legend()
            fig_all_folds_train_only.savefig(
                save_path + "{}_{}_train.png".format(metric, specific_trainer[0].model_name))
            fig_all_folds_train_only.clf()

            ax_all_folds_val_only.set_title(metric)
            ax_all_folds_val_only.legend()
            fig_all_folds_val_only.savefig(
                save_path + "{}_{}_val.png".format(metric, specific_trainer[0].model_name))
            fig_all_folds_val_only.clf()

        ax_all_trainers.set_title(metric)
        ax_all_trainers.legend()
        fig_all_trainers.savefig(save_path + "{}.png".format(metric))
        fig_all_trainers.clf()

        ax_all_trainers_test_only.set_title(metric)
        ax_all_trainers_test_only.legend()
        fig_all_trainers_test_only.savefig(save_path + "{}_test.png".format(metric))
        fig_all_trainers_test_only.clf()

        ax_all_trainers_train_only.set_title(metric)
        ax_all_trainers_train_only.legend()
        fig_all_trainers_train_only.savefig(save_path + "{}_train.png".format(metric))
        fig_all_trainers_train_only.clf()

        ax_all_trainers_val_only.set_title(metric)
        ax_all_trainers_val_only.legend()
        fig_all_trainers_val_only.savefig(save_path + "{}_val.png".format(metric))
        fig_all_trainers_val_only.clf()


