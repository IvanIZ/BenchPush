"""
An example script for training and evaluating baselines for the maze NAMO task
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import argparse
import pickle
from benchpush.baselines.maze_NAMO.ppo.policy import MazeNAMOPPO
from benchpush.baselines.maze_NAMO.sac.policy import MazeNAMOSAC
from benchpush.common.metrics.base_metric import BaseMetric
from benchpush.common.utils.utils import DotDict
from os.path import dirname

def main(cfg, job_id):

    if cfg.train.train_mode:

        model_name = cfg.train.job_name

        if cfg.train.job_type == 'ppo':
            # ================================ PPO Policy =================================    
            ppo_policy = MazeNAMOPPO(model_name=model_name, cfg=cfg)
            ppo_policy.train(total_timesteps=cfg.train.total_timesteps, checkpoint_freq=cfg.train.checkpoint_freq)

        elif cfg.train.job_type == 'sac':
            # ================================ SAC Policy =================================
            sac_policy = MazeNAMOSAC(model_name=model_name, cfg=cfg)
            sac_policy.train(total_timesteps=cfg.train.total_timesteps, checkpoint_freq=cfg.train.checkpoint_freq)

    if cfg.evaluate.eval_mode:
        benchmark_results = []
        num_eps = cfg.evaluate.num_eps
        model_path = cfg.evaluate.model_path
        for policy_type, model_name in zip(cfg.evaluate.policy_types, cfg.evaluate.model_names):
            cfg.train.job_type = policy_type
            cfg.maze_version = cfg.evaluate.maze_version

            if policy_type == 'ppo':
                # ================================ PPO Policy =================================    
                ppo_policy = MazeNAMOPPO(model_name=model_name, model_path=model_path, cfg=cfg)
                benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps))

            elif policy_type == 'sac':
                # ================================ SAC Policy =================================
                sac_policy = MazeNAMOSAC(model_name=model_name, model_path=model_path, cfg=cfg)
                benchmark_results.append(sac_policy.evaluate(num_eps=num_eps))
        
        BaseMetric.plot_algs_scores(benchmark_results, save_fig_dir='./')

        # save eval results to disk
        pickle_dict = {
            'benchmark_results': benchmark_results
        }
        with open('shipIce_benchmark_results.pkl', 'wb') as f:
                pickle.dump(pickle_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate baselines for maze navigation'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        help='path to the config file',
        default=None
    )

    parser.add_argument(
        '--job_id',
        type=str,
        help='slurm job id',
        default=None
    )

    job_id = parser.parse_args().job_id

    if parser.parse_args().config_file is not None:
        cfg = DotDict.load_from_file(parser.parse_args().config_file)


    else:
        # High level configuration for the box delivery task
        cfg = {
            'output_dir': "logs/",  # 'output/trial0'
            'obstacles': {
            'num_obstacles': 5,
            'obstacle_size': 0.5,
            'maze_version': 1, # options are 1, 2
            },
            'train': {
                'train_mode': False,
                'job_type': 'ppo', # 'ppo', 'sac'
                'job_name': 'maze_ppo',
                'total_timesteps': int(15e5),
                'checkpoint_freq': 10000,
            },
            'evaluate': {
                'eval_mode': True,
                'num_eps': 1,
                'policy_types': ['ppo', 'sac'], # list of policy types to evaluate
                'model_names': ['ppo_model', 'sac_model'], # list of model names to evaluate
                'model_path': 'models/maze', # path to the models
                'maze_version': 1, # list of maze versions to evaluate
            },
        }

        cfg = DotDict.to_dot_dict(cfg)

    main(cfg, job_id)

