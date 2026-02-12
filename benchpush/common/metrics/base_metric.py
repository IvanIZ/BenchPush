from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import os
from typing import List, Tuple


class BaseMetric(ABC):
    """
    A base metric class
    """

    def __init__(self, alg_name) -> None:
        self.rewards = []
        self.efficiency_scores = []
        self.effort_scores = []
        self.success_rates = []

        self.alg_name = alg_name

    
    def plot_scores(self, save_fig_dir):
        """
        Generate box plots for efficiency scores, effort scores, and rewards on a single algorithm
        """

        fig, ax = plt.subplots()

        ax.clear()
        bp_data = [self.efficiency_scores]
        ax.boxplot(bp_data, showmeans=True)
        ax.set_title("Efficiency Plot")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Efficiency Scores")
        fp = os.path.join(save_fig_dir, self.alg_name + '_efficiency.png')
        fig.savefig(fp)

        ax.clear()
        bp_data = [self.effort_scores]
        ax.boxplot(bp_data, showmeans=True)
        ax.set_title("Effort Plot")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Effort Scores")
        fp = os.path.join(save_fig_dir, self.alg_name + '_effort.png')
        fig.savefig(fp)

        ax.clear()
        bp_data = [self.rewards]
        ax.boxplot(bp_data, showmeans=True)
        ax.set_title("Rewards Plot")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Rewards")
        fp = os.path.join(save_fig_dir, self.alg_name + '_rewards.png')
        fig.savefig(fp)

        ax.clear()
        bp_data = [self.success_rates]
        ax.boxplot(bp_data, showmeans=True)
        ax.set_title("Success Rates Plot")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Success Rates")
        fp = os.path.join(save_fig_dir, self.alg_name + '_success_rates.png')
        fig.savefig(fp)

        plt.close('all')


    @staticmethod
    def plot_algs_score(scores, score_name, alg_names, save_fig_dir, filename, legend=True):
        fig, ax = plt.subplots()

        color_list = [
            (0.43, 0.64, 0.68), 
            (0.84, 0.39, 0.26),
            (0.65, 0.65, 0.65),
            (0.3, 0.3, 0.3), 
            (0.1, 0.1, 0.1), 
        ]

        # plot scores
        ax.clear()
        bps = []
        colors = []
        positions = []
        for i in range(len(scores)):
            score = scores[i]

            color = color_list[i]
            position = 1.5 * i + 1
            bp = ax.boxplot([score], positions=[position], showmeans=False, widths=0.8,
            patch_artist=True, boxprops=dict(facecolor=color), medianprops=dict(color="black"))

            bps.append(bp["boxes"][0])
            colors.append(color)
            positions.append(position)

        ax.set_xticks(positions)
        ax.set_xticklabels(alg_names)
        # ax.set_xlabel(score_name)
        # ax.set_ylabel("Algorithms")
        # ax.legend(bps, alg_names, loc="upper left")
        ax.legend(bps, alg_names, loc="lower right")
        ax.set_xlim(0, 1.5 * len(scores) + 0.5)
        fp = os.path.join(save_fig_dir, filename + '.png')
        fig.savefig(fp)


    @staticmethod
    def plot_algs_scores(benchmark_results, save_fig_dir: str, plot_success=False) -> None:
        """
        :param benchmark_results: a list of evaluation tuples, where each tuple is computed from policy.evaluate()
        """

        # parse benchmark results
        efficiency_data = []
        effort_data = []
        reward_data = []
        alg_names = []
        success_data = []
        if plot_success:
            for alg_success, alg_efficiency, alg_effort, alg_reward, alg_name in benchmark_results:
                efficiency_data.append(alg_efficiency)
                effort_data.append(alg_effort)
                reward_data.append(alg_reward)
                alg_names.append(alg_name)
                success_data.append(alg_success)
        else:
            for alg_efficiency, alg_effort, alg_reward, alg_name in benchmark_results:
                efficiency_data.append(alg_efficiency)
                effort_data.append(alg_effort)
                reward_data.append(alg_reward)
                alg_names.append(alg_name)

        BaseMetric.plot_algs_score(scores=efficiency_data, score_name="Efficiency Score", alg_names=alg_names, save_fig_dir=save_fig_dir, filename="efficiency_benchmark")
        BaseMetric.plot_algs_score(scores=effort_data, score_name="Effort Score", alg_names=alg_names, save_fig_dir=save_fig_dir, filename="effort_benchmark")
        BaseMetric.plot_algs_score(scores=reward_data, score_name="Rewards", alg_names=alg_names, save_fig_dir=save_fig_dir, filename="reward_benchmark")
        if plot_success:
            BaseMetric.plot_algs_score(scores=success_data, score_name="Task Success Score", alg_names=alg_names, save_fig_dir=save_fig_dir, filename="success_benchmark")

    @staticmethod
    def plot_algs_scores_task_driven(benchmark_results: List[Tuple[List[float], List[float], List[float], List[float], str]], save_fig_dir: str) -> None:
        """
        :param benchmark_results: a list of evaluation tuples, where each tuple is computed from policy.evaluate()
        """

        # parse benchmark results
        success_data = []
        efficiency_data = []
        effort_data = []
        reward_data = []
        alg_names = []
        for alg_success, alg_efficiency, alg_effort, alg_reward, alg_name in benchmark_results:
            success_data.append(alg_success)
            efficiency_data.append(alg_efficiency)
            effort_data.append(alg_effort)
            reward_data.append(alg_reward)
            alg_names.append(alg_name)

        BaseMetric.plot_algs_score(scores=success_data, score_name="Task Success Score", alg_names=alg_names, save_fig_dir=save_fig_dir, filename="success_rate_benchmark")
        BaseMetric.plot_algs_score(scores=efficiency_data, score_name="Efficiency Score", alg_names=alg_names, save_fig_dir=save_fig_dir, filename="efficiency_benchmark")
        BaseMetric.plot_algs_score(scores=effort_data, score_name="Effort Score", alg_names=alg_names, save_fig_dir=save_fig_dir, filename="effort_benchmark")
        BaseMetric.plot_algs_score(scores=reward_data, score_name="Rewards", alg_names=alg_names, save_fig_dir=save_fig_dir, filename="reward_benchmark")


    @abstractmethod
    def compute_efficiency_score(self):
        """
        Implement this function to compute efficiency score for a trial
        """
        raise NotImplementedError


    @abstractmethod
    def compute_effort_score(self):
        """
        Implement this function to compute interaction effort score for a trial
        """
        raise NotImplementedError


    @abstractmethod
    def update(self, info, reward, eps_complete=False):
        """
        Implement this function for any accumulative metrics
        """
        raise NotImplementedError


    @abstractmethod
    def reset(self, info):
        """
        Implement this function to reset any trial-specific values upon starting a new trial
        """
        raise NotImplementedError
    