"""
An example script for training and evaluating baselines for ship ice navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import argparse
from benchpush.baselines.ship_ice_nav.ppo.policy import ShipIcePPO
from benchpush.baselines.ship_ice_nav.sac.policy import ShipIceSAC
from benchpush.baselines.ship_ice_nav.planning_based.policy import PlanningBasedPolicy
from benchpush.common.metrics.base_metric import BaseMetric
from benchpush.common.utils.utils import DotDict
from os.path import dirname
import pickle

def main(cfg, job_id):

    if cfg.train.train_mode:

        model_name = cfg.train.job_name

        if cfg.train.job_type == 'ppo':
            # ================================ PPO Policy =================================    
            ppo_policy = ShipIcePPO(model_name=model_name, cfg=cfg)
            ppo_policy.train(total_timesteps=cfg.train.total_timesteps, checkpoint_freq=cfg.train.checkpoint_freq)

        elif cfg.train.job_type == 'sac':
            # ================================ SAC Policy =================================
            sac_policy = ShipIceSAC(model_name=model_name, cfg=cfg)
            sac_policy.train(total_timesteps=cfg.train.total_timesteps, checkpoint_freq=cfg.train.checkpoint_freq)

    if cfg.evaluate.eval_mode:
        benchmark_results = []
        num_eps = cfg.evaluate.num_eps
        model_path = cfg.evaluate.model_path
        for policy_type, observation_type, model_name in zip(cfg.evaluate.policy_types, cfg.evaluate.observation_types, cfg.evaluate.model_names):
            cfg.train.job_type = policy_type
            cfg.egocentric_obs = observation_type

            if policy_type == 'ppo':
                # ================================ PPO Policy =================================    
                ppo_policy = ShipIcePPO(model_name=model_name, model_path=model_path, cfg=cfg)
                benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps))

            elif policy_type == 'sac':
                # ================================ SAC Policy =================================
                sac_policy = ShipIceSAC(model_name=model_name, model_path=model_path, cfg=cfg)
                benchmark_results.append(sac_policy.evaluate(num_eps=num_eps))

            elif policy_type == 'lattice':
                # ================================ Lattice Planning Policy =================================
                lattice_planning_policy = PlanningBasedPolicy(planner_type='lattice', cfg=cfg)
                benchmark_results.append(lattice_planning_policy.evaluate(num_eps=num_eps))

            elif policy_type == 'predictive':
                # ================================ Predictive Planning Policy =================================
                predictive_planning_policy = PlanningBasedPolicy(planner_type='predictive', cfg=cfg)
                benchmark_results.append(predictive_planning_policy.evaluate(num_eps=num_eps))
        
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
            'output_dir': 'logs/',
            'egocentric_obs': False,
            'concentration': 0.1,  # options are 0.1, 0.2, 0.3, 0.4, 0.5
            'goal_y': 9,
            'train': {
                'train_mode': False,
                'job_type': 'ppo', # 'ppo', 'sac'
                'job_name': 'shipice_ppo',
                'total_timesteps': int(15e5),
                'checkpoint_freq': 10000,
            },
            'evaluate': {
                'eval_mode': True,
                'num_eps': 1,
                'policy_types': ['ppo', 'sac', 'lattice', 'predictive'],    # list of policy types to evaluate
                'observation_types': [True, True, False, False],               # RL policies use egocentric observations
                'model_names': ['ppo_model', 'sac_model', '', ''],          # list of model names to evaluate
                'model_path': './models/ship_ice', # path to the models
            },
        }

        cfg = DotDict.to_dot_dict(cfg)

    main(cfg, job_id)

