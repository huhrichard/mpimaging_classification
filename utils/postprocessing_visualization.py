from utils.common_library import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def compare_model(trainers, save_path):

    n = len(trainers)
    colors = plt.cm.jet(np.linspace(0, 1, n))
    maxepochs = 0
    plt.clf()
    metrics = list(trainers[0][0].performance_stat["train"][0].keys())
    epochs = trainers[0][0].total_epochs
    states = ["train", "val", "test"]
    linestyle_dict = {"train": "-",
                      "val": ".",
                      "test": "o"}
    for metric in metrics:
        fig_all_trainers = plt.figure()
        ax_all_trainers = fig_all_trainers.add_subplot(1,1,1)
        fig_all_trainers_test_only = plt.figure()
        ax_all_trainers_test_only = fig_all_trainers_test_only.add_subplot(1, 1, 1)
        for idx, trainer in enumerate(trainers):
            # plot each fold result 1st
            fig_all_folds = plt.figure()
            ax_all_folds = fig_all_folds.add_subplot(1,1,1)
            fig_all_folds_test_only = plt.figure()
            ax_all_folds_test_only = fig_all_folds_test_only.add_subplot(1,1,1)

            metric_dicts = {"train":[],
                            "val":[],
                            "test":[]
                            }
            for nth_fold in range(len(trainer)):
                nth_fold_trainer = trainer[nth_fold]
                fig_all_states = plt.figure()
                ax_all_states = fig_all_states.add_subplot(1,1,1)

                for state in states:
                    fig_temp = plt.figure()
                    ax_temp = fig_temp.add_subplot(1,1,1)
                    metric_scores = [nth_fold_trainer[state][e][metric] for e in epochs]

                    metric_dicts[state].append(metric_scores)
                    # stacked_fold_metrics.append(metric_scores)

                    plot_paras = {"x":range(epochs),
                                  "y":metric_scores,
                                  "label":"{}({})".format(state, nth_fold_trainer.model_name),
                                  "c":colors[idx],
                                  "linestyle":linestyle_dict[state],
                                  "alpha": 0.7}

                    ax_temp.plot(**plot_paras)
                    ax_all_states.plot(**plot_paras)

                    ax_temp.set_title(metric)
                    ax_temp.legend()
                    fig_temp.savefig(save_path+"{}_{}thfold_{}_{}.png".format(metric, nth_fold_trainer.model_name,
                                                                           nth_fold, state))

                ax_all_states.set_title(metric)
                ax_all_states.legend()
                fig_all_states.savefig(save_path+"{}_{}thfold_{}.png".format(metric, nth_fold_trainer.model_name,
                                                                           nth_fold))

            for state in states:
                metric_scores[state] = np.array(metric_scores[state])
                plot_paras = {"x": range(epochs),
                              "y": np.mean(metric_scores[state], axis=0),
                              "yerr": np.std(metric_scores[state], axis=0),
                              "label": "{}({})".format(state, trainer[0].model_name),
                              "c": colors[idx],
                              "linestyle": linestyle_dict[state],
                              "alpha": 0.7}
                if state == "test":
                    ax_all_folds_test_only.errorbar(**plot_paras)
                    ax_all_trainers_test_only.errorbar(**plot_paras)
                ax_all_folds.errorbar(**plot_paras)
                ax_all_trainers.errorbar(**plot_paras)

            ax_all_folds.set_title(metric)
            ax_all_folds.legend()
            fig_all_folds.savefig(save_path + "{}_{}.png".format(metric, trainer[0].model_name))

            ax_all_folds_test_only.set_title(metric)
            ax_all_folds_test_only.legend()
            fig_all_folds_test_only.savefig(save_path + "{}_{}_testing.png".format(metric, trainer[0].model_name))

        ax_all_trainers.set_title(metric)
        ax_all_trainers.legend()
        fig_all_trainers.savefig(save_path + "{}.png".format(metric))

        ax_all_trainers_test_only.set_title(metric)
        ax_all_trainers_test_only.legend()
        fig_all_trainers_test_only.savefig(save_path + "{}_testing.png".format(metric))



