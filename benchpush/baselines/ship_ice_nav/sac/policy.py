from benchpush.baselines.base_class import BasePolicy
from benchpush.baselines.feature_extractors import ResNet18
from benchpush.common.metrics.ship_ice_metric import ShipIceMetric
from typing import List, Tuple
import benchpush.environments
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import numpy as np


class ShipIceSAC(BasePolicy):

    def __init__(self, model_name='sac_model', model_path=None, cfg=None) -> None:
        super().__init__()

        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), 'models/')
        else:
            self.model_path = model_path

        self.model_name = model_name
        self.model = None

        self.cfg = cfg


    def train(self, policy_kwargs=dict(features_extractor_class=ResNet18,
                                        features_extractor_kwargs=dict(features_dim=512),
                                        net_arch=[512, 256]),
            batch_size=128,
            buffer_size=15000,
            learning_starts=200,
            learning_rate=5e-4,
            gamma=0.97,
            verbose=2,
            total_timesteps=int(2e5), 
            checkpoint_freq=10000) -> None:

        env = gym.make('ship-ice-v0', cfg=self.cfg)
        env = env.unwrapped

        # The noise objects for SAC
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))

        self.model = SAC(
            'CnnPolicy', 
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            action_noise=action_noise,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=1,
            gradient_steps=1,
            verbose=verbose,
            tensorboard_log=self.model_path)

        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=self.model_path,
            name_prefix=self.model_name,
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        # Train and save the agent
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        self.model.save(os.path.join(self.model_path, self.model_name))
        env.close()



    def evaluate(self, num_eps: int, model_eps: str ='latest') -> Tuple[List[float], List[float], List[float], str]:

        if model_eps == 'latest':
            self.model = SAC.load(os.path.join(self.model_path, self.model_name))
        else:
            model_checkpoint = self.model_name + '_' + model_eps + '_steps'
            self.model = SAC.load(os.path.join(self.model_path, model_checkpoint))

        env = gym.make('ship-ice-v0', cfg=self.cfg)
        env = env.unwrapped
        metric = ShipIceMetric(alg_name="SAC", ship_mass=env.cfg.ship.mass, goal=env.goal)

        for eps_idx in range(num_eps):
            print("SAC Progress: ", eps_idx, " / ", num_eps, " episodes")
            obs, info = env.reset()
            metric.reset(info)
            done = truncated = False
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                metric.update(info=info, reward=reward, eps_complete=(done or truncated))
                if done or truncated:
                    break
        
        env.close()
        metric.plot_scores(save_fig_dir=env.cfg.output_dir)
        return metric.efficiency_scores, metric.effort_scores, metric.rewards, "SAC"


    
    def act(self, observation, **kwargs):

        # parameters for planners
        model_eps = kwargs.get('model_eps', None)
        
        # load trained model for the first time
        if self.model is None:

            if model_eps is None:
                self.model = SAC.load(os.path.join(self.model_path, self.model_name))
            else:
                model_checkpoint = self.model_name + '_' + model_eps + '_steps'
                self.model = SAC.load(os.path.join(self.model_path, model_checkpoint))

        action, _ = self.model.predict(observation, deterministic=True)
        return action
