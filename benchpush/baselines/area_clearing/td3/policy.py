from benchpush.baselines.base_class import BasePolicy
import benchpush.environments
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import numpy as np


class AreaClearingTD3(BasePolicy):

    def __init__(self, model_name='td3_model', model_path=None) -> None:
        super().__init__()

        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), 'models/')
        else:
            self.model_path = model_path

        self.model_name = model_name
        self.model = None


    def train(self, policy_kwargs=dict(net_arch=[256, 256]),
            batch_size=64,
            buffer_size=15000,
            learning_starts=200,
            learning_rate=5e-4,
            gamma=0.97,
            verbose=2,
            total_timesteps=int(2e5), 
            checkpoint_freq=100) -> None:

        env = gym.make('area-clearing-v0')
        env = env.unwrapped

        print("Checking environment")
        check_env(env)

        # The noise objects for TD3
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        self.model = TD3('CnnPolicy', env,
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



    def evaluate(self, num_eps: int, model_eps: str ='latest'):

        if model_eps == 'latest':
            self.model = TD3.load(os.path.join(self.model_path, self.model_name))
        else:
            model_checkpoint = self.model_name + '_' + model_eps + '_steps'
            self.model = TD3.load(os.path.join(self.model_path, model_checkpoint))

        env = gym.make('area-clearing-v0')
        env = env.unwrapped

        rewards_list = []
        for eps_idx in range(num_eps):
            print("Progress: ", eps_idx, " / ", num_eps, " episodes")
            obs, info = env.reset()
            done = truncated = False
            eps_reward = 0.0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                eps_reward += reward
                if done or truncated:
                    rewards_list.append(eps_reward)
                    break
        
        env.close()
        return rewards_list


    
    def act(self, observation, **kwargs):
        
        # load trained model for the first time
        if self.model is None:
            self.model = TD3.load(os.path.join(self.model_path, self.model_name))

        action, _ = self.model.predict(observation, deterministic=True)
        return action
