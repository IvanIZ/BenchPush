"""
An example script for training and evaluating baselines for box pushing navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import argparse
from benchpush.baselines.box_delivery.SAM.policy import BoxDeliverySAM
from benchpush.baselines.box_delivery.ppo.policy import BoxDeliveryPPO
from benchpush.baselines.box_delivery.sac.policy import BoxDeliverySAC
from benchpush.common.metrics.base_metric import BaseMetric
from benchpush.common.utils.utils import DotDict
from os.path import dirname

def main(cfg, job_id):

    if cfg.train.train_mode:
        if cfg.train.resume_training:
            model_name = cfg.train.job_id_to_resume
        else:
            model_name = f'{cfg.train.job_name}_{job_id}'

        if cfg.train.job_type == 'sam':
            # ========================= Spatial Action Map Policy =========================
            sam_policy = BoxDeliverySAM(model_name=model_name, cfg=cfg)
            sam_policy.train(job_id)

        elif cfg.train.job_type == 'ppo':
            # ================================ PPO Policy =================================    
            ppo_policy = BoxDeliveryPPO(model_name=model_name, cfg=cfg)
            ppo_policy.train(resume_training=cfg.train.resume_training, n_steps=cfg.train.n_steps, batch_size=cfg.train.batch_size)

        elif cfg.train.job_type == 'sac':
            # ================================ SAC Policy =================================
            sac_policy = BoxDeliverySAC(model_name=model_name, cfg=cfg)
            sac_policy.train(resume_training=cfg.train.resume_training, batch_size=cfg.train.batch_size, learning_starts=cfg.train.learning_starts)
    
    if cfg.evaluate.eval_mode:
        benchmark_results = []
        num_eps = cfg.evaluate.num_eps
        model_path = cfg.evaluate.model_path
        for policy_type, action_type, model_name, obs_config in zip(cfg.evaluate.policy_types, cfg.evaluate.action_types, cfg.evaluate.model_names, cfg.evaluate.obs_configs):
            cfg.agent.action_type = action_type
            cfg.train.job_type = policy_type
            cfg.env.obstacle_config = obs_config

            if policy_type == 'sam':
                # ========================= Spatial Action Map Policy =========================
                sam_policy = BoxDeliverySAM(model_name=model_name, model_path=model_path, cfg=cfg)
                benchmark_results.append(sam_policy.evaluate(num_eps=num_eps))

            elif policy_type == 'ppo':
                # ================================ PPO Policy =================================    
                ppo_policy = BoxDeliveryPPO(model_name=model_name, model_path=model_path, cfg=cfg)
                benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps))

            elif policy_type == 'sac':
                # ================================ SAC Policy =================================
                sac_policy = BoxDeliverySAC(model_name=model_name, model_path=model_path, cfg=cfg)
                benchmark_results.append(sac_policy.evaluate(num_eps=num_eps))

        BaseMetric.plot_algs_scores(benchmark_results, save_fig_dir='./', plot_success=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate baselines for box pushing task'
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
        cfg={
            'render': {
                'show': True,           # if true display the environment
                'show_obs': False,       # if true show observation
            },
            'agent': {
                'action_type': 'position', # 'position', 'heading', 'velocity'
            },
            'boxes': {
                'num_boxes_small': 10,
                'num_boxes_large': 20,
            },
            'env': {
                'obstacle_config': 'small_empty', # options are small_empty, small_columns, large_columns, large_divider
            },
            'train': {
                'train_mode': False,
                'job_type': 'sam', # 'sam', 'ppo', 'sac'
                'job_name': 'SAM',
                'resume_training': False,
                'job_id_to_resume': None,
            },
            'evaluate': {
                'eval_mode': True,
                'num_eps': 1,
                'policy_types': ['ppo', 'sac', 'sam'], # list of policy types to evaluate
                'action_types': ['heading', 'heading', 'position'], # list of action types to evaluate
                'model_names': ['ppo_small_empty', 'sac_small_empty', 'sam_small_empty'], # list of model names to evaluate
                'model_path': 'models/box_delivery', # path to the models
                'obs_configs': ['small_empty', 'small_empty', 'small_empty'], # list of observation configurations
            }
        }
        
        cfg = DotDict.to_dot_dict(cfg)

    main(cfg, job_id)
