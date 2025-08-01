from benchnpin.baselines.base_class import BasePolicy
from benchnpin.baselines.feature_extractors import DenseActionSpaceDQN
from benchnpin.common.metrics.task_driven_metric import TaskDrivenMetric
from benchnpin.common.utils.utils import DotDict
import benchnpin.environments
import gymnasium as gym
from collections import namedtuple
import random
import os
import sys
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import logging

logging.getLogger('pymunk').propagate = False


# enable cuDNN auto-tuner to find the best algorithm to use for your hardware
torch.backends.cudnn.benchmark = True

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'ministeps', 'next_state'))

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Meters:
    def __init__(self):
        self.meters = {}

    def get_names(self):
        return self.meters.keys()

    def reset(self):
        for _, meter in self.meters.items():
            meter.reset()

    def update(self, name, val):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(val)

    def avg(self, name):
        return self.meters[name].avg

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)
    
class DenseActionSpacePolicy:
    def __init__(self, action_space, num_input_channels, final_exploration, train=False, checkpoint_path='', resume_training=False, evaluate=False, job_id_to_resume=None, random_seed=None, model_name='sam_model', model_path=None):
        self.action_space = action_space
        self.num_input_channels = num_input_channels
        self.final_exploration = final_exploration
        self.train = train

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = self.build_network()
        self.transform = transforms.ToTensor()

        # Resume from checkpoint if applicable
        if os.path.exists(checkpoint_path) or resume_training or evaluate:
            if resume_training:
                model_path = os.path.join(os.path.dirname(__file__), f'checkpoint/{job_id_to_resume}/model-{model_name}.pt')
            elif evaluate:
                if model_path is not None:
                    model_path = model_path + f'/{model_name}.pt'
                else:
                    model_path = os.path.join(os.path.dirname(__file__), f'models/{model_name}.pt')
            else:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                model_path = f'{checkpoint_dir}/model-{self.model_name}.pt'
            model_checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(model_checkpoint['state_dict'])
            if self.train:
                self.policy_net.train()
            else:
                self.policy_net.eval()
            print(f"=> loaded model '{model_path}'")
            logging.info(f"=> loaded model '{model_path}'")

        if random_seed is not None:
            random.seed(random_seed)

    def build_network(self):
        return torch.nn.DataParallel(
            DenseActionSpaceDQN(num_input_channels=self.num_input_channels)
        ).to(self.device)

    def apply_transform(self, s):
        return self.transform(s).unsqueeze(0)

    def step(self, state, exploration_eps=None, debug=False):
        if exploration_eps is None:
            exploration_eps = self.final_exploration
        state = self.apply_transform(state).to(self.device)
        with torch.no_grad():
            output = self.policy_net(state).squeeze(0)
        if random.random() < exploration_eps:
            action = random.randrange(self.action_space)
        else:
            action = output.view(1, -1).max(1)[1].item()
        info = {}
        if debug:
            info['output'] = output.squeeze(0)
        return action, info

class AreaClearingSAM(BasePolicy):

    def __init__(self, model_name='sam_model', model_path=None, cfg=None) -> None:
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), 'models/')
        else:
            self.model_path = model_path

        self.model_name = model_name
        self.model = None

        self.cfg = cfg


    def update_policy(self, policy_net, target_net, optimizer, batch, transform_func):
        state_batch = torch.cat([transform_func(s) for s in batch.state]).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        ministeps_batch = torch.tensor(batch.ministeps, dtype=torch.float32).to(self.device)
        non_final_next_states = torch.cat([transform_func(s) for s in batch.next_state if s is not None]).to(self.device, non_blocking=True)

        output = policy_net(state_batch)
        state_action_values = output.view(self.batch_size, -1).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_state_values = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=self.device)

        if self.use_double_dqn:
            with torch.no_grad():
                best_action = policy_net(non_final_next_states).view(non_final_next_states.size(0), -1).max(1)[1].view(non_final_next_states.size(0), 1)
                next_state_values[non_final_mask] = target_net(non_final_next_states).view(non_final_next_states.size(0), -1).gather(1, best_action).view(-1)
        else:
            next_state_values[non_final_mask] = target_net(non_final_next_states).view(non_final_next_states.size(0), -1).max(1)[0].detach()

        expected_state_action_values = (reward_batch + torch.pow(self.gamma, ministeps_batch) * next_state_values)
        td_error = torch.abs(state_action_values - expected_state_action_values).detach()
        loss = smooth_l1_loss(state_action_values, expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), self.grad_norm_clipping)
        optimizer.step()

        train_info = {}
        train_info['q_value_min'] = output.min().item()
        train_info['q_value_max'] = output.max().item()
        train_info['td_error'] = td_error.mean()
        train_info['loss'] = loss

        return train_info


    def train(self, job_id) -> None:
        # create environment
        env = gym.make('area-clearing-v0', cfg=self.cfg)
        env = env.unwrapped
        env.configure_env_for_SAM()
        self.cfg = env.cfg # update cfg with env-specific config

        job_id = job_id
        params = self.cfg['train']
        self.batch_size = params['batch_size']
        self.checkpoint_freq = params['checkpoint_freq']
        self.final_exploration = params['final_exploration']
        self.gamma = params['gamma']
        self.grad_norm_clipping = params['grad_norm_clipping']
        self.learning_rate = params['learning_rate']
        self.replay_buffer_size = params['replay_buffer_size']
        self.weight_decay = params['weight_decay']

        checkpoint_freq = params['checkpoint_freq']
        exploration_timesteps = params['exploration_timesteps']
        job_id_to_resume = params['job_id_to_resume']
        learning_starts = params['learning_starts']
        resume_training = params['resume_training']
        target_update_freq = params['target_update_freq']
        total_timesteps = params['total_timesteps']

        checkpoint_path = os.path.join(os.path.dirname(__file__), f'checkpoint/{job_id}/checkpoint-{self.model_name}.pt')

        log_dir = os.path.join(os.path.dirname(__file__), 'output_logs/')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logging.basicConfig(filename=os.path.join(log_dir, f'{self.model_name}.log'), level=logging.DEBUG)
        logging.info("starting training...")
        logging.info(f"Job ID: {job_id}")

        # policy
        policy = DenseActionSpacePolicy(env.action_space.high, env.num_channels, self.final_exploration,
                                         train=True, checkpoint_path=checkpoint_path, resume_training=resume_training)

        # optimizer
        optimizer = optim.SGD(policy.policy_net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)

        # replay buffer
        replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # resume if possible
        start_timestep = 0
        episode = 0
        if os.path.exists(checkpoint_path) or resume_training:
            if resume_training:
                checkpoint_path_to_load = os.path.join(os.path.dirname(__file__), f'checkpoint/{job_id_to_resume}/checkpoint-{self.model_name}.pt')
            else:
                checkpoint_path_to_load = checkpoint_path
            checkpoint = torch.load(checkpoint_path_to_load)
            start_timestep = checkpoint['timestep']
            episode = checkpoint['episode']
            optimizer.load_state_dict(checkpoint['optimizer'])
            replay_buffer = checkpoint['replay_buffer']
            print(f"=> loaded checkpoint '{checkpoint_path}' (timestep: {start_timestep})")
            logging.info(f"=> loaded checkpoint '{checkpoint_path}' (timestep: {start_timestep})")
        else:
            logging.info("=> no checkpoint detected, starting from initial state")
        
        # target net
        target_net = policy.build_network()
        target_net.load_state_dict(policy.policy_net.state_dict())
        target_net.eval()

        # logging
        train_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, f'{job_id}'))
        meters = Meters()

        state, _ = env.reset()
        total_timesteps_with_warmup = total_timesteps + learning_starts
        for timestep in tqdm(range(start_timestep, total_timesteps_with_warmup),
                             initial=start_timestep, total=total_timesteps_with_warmup, file=sys.stdout):
            
            start_time = time.time()

            # select action
            if exploration_timesteps > 0:
                exploration_eps = 1 - min(max(timestep - learning_starts, 0) / exploration_timesteps, 1) * (1 - self.final_exploration)
            else:
                exploration_eps = self.final_exploration
            action, _ = policy.step(state, exploration_eps=exploration_eps)

            # step the simulation
            next_state, reward, done, truncated, info = env.step(action)
            ministeps = info['ministeps']

            # store in buffer
            replay_buffer.push(state, action, reward, ministeps, next_state)
            state = next_state

            # reset if episode ended
            if done:
                state, _ = env.reset()
                episode += 1
                if truncated:
                    logging.info(f"Episode {episode} truncated. {info['cumulative_cubes']} in goal. Resetting environment...")
                else:
                    logging.info(f"Episode {episode} completed. Resetting environment...")
            
            # train network
            if timestep >= learning_starts:
                batch = replay_buffer.sample(self.batch_size)
                train_info = self.update_policy(policy.policy_net, target_net, optimizer, batch, policy.apply_transform)
            
            # update target network
            if (timestep + 1) % target_update_freq == 0:
                target_net.load_state_dict(policy.policy_net.state_dict())
            
            step_time = time.time() - start_time

            ################################################################################
            # Logging
            # meters
            # meters.update('step_time', step_time)
            # if timestep >= learning_starts:
            #     for name, val in train_info.items():
            #         meters.update(name, val)
            
            # if done:
            #     for name in meters.get_names():
            #         train_summary_writer.add_scalar(name, meters.avg(name), timestep + 1)
            #     eta_seconds = meters.avg('step_time') * (total_timesteps_with_warmup - timestep)
            #     meters.reset()

            #     train_summary_writer.add_scalar('episodes', episode, timestep + 1)
            #     train_summary_writer.add_scalar('eta_hours', eta_seconds / 3600, timestep + 1)

            #     for name in ['cumulative_cubes', 'cumulative_distance', 'cumulative_reward']:
            #         train_summary_writer.add_scalar(name, info[name], timestep + 1)

            ################################################################################
            # Checkpoint
            if (timestep + 1) % checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warmup:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                model_path = f'{checkpoint_dir}/model-{self.model_name+str(timestep+1)}.pt'
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                # temp_model_path = f'{checkpoint_dir}/model-temp.pt'
                model = {
                    'timestep': timestep + 1,
                    'state_dict': policy.policy_net.state_dict(),
                }

                temp_checkpoint_path = f'{checkpoint_dir}/checkpoint-temp.pt'
                checkpoint = {
                    'timestep': timestep + 1,
                    'episode': episode,
                    'optimizer': optimizer.state_dict(),
                    'replay_buffer': replay_buffer,
                }

                # save model and checkpoint
                torch.save(model, model_path)
                torch.save(checkpoint, temp_checkpoint_path)

                # according to the GNU spec of rename, the state of checkpoint_path
                # is atomic, i.e. it will either be modified or not modified, but not in
                # between, during a system crash (i.e. preemtion)
                # os.replace(temp_model_path, model_path)
                os.replace(temp_checkpoint_path, checkpoint_path)
                msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + checkpoint_path + self.model_name
                logging.info(msg)
        env.close()


    def evaluate(self, num_eps: int, model_eps: str ='latest'):
        # create environment
        if self.cfg is not None:
            env = gym.make('area-clearing-v0', cfg=self.cfg)
            # env = gym.make('area-clearing-v0')
        else:
            env = gym.make('area-clearing-v0')
        env = env.unwrapped

        env.configure_env_for_SAM()

        old_action_type = env.cfg.agent.action_type
        old_t_max = env.cfg.sim.t_max

        env.cfg.agent.action_type = 'position'
        env.cfg.sim.t_max = 50

        metric = TaskDrivenMetric(alg_name="SAM", robot_mass=env.cfg.agent.mass)

        if model_eps == 'latest':
            self.model = DenseActionSpacePolicy(env.action_space.high, env.num_channels, 0.0,
                                                train=False, evaluate=True, model_name=self.model_name, model_path=self.model_path)
        else:
            model_checkpoint = self.model_name + '_' + model_eps + '_steps'

        rewards_list = []
        for eps_idx in range(num_eps):
            print("SAM Progress: ", eps_idx, " / ", num_eps, " episodes")
            obs, info = env.reset()
            metric.reset(info)
            done = truncated = False
            eps_reward = 0.0
            while True:
                action, _ = self.model.step(obs)
                obs, reward, done, truncated, info = env.step(action)
                metric.update(info=info, reward=reward, eps_complete=(done or truncated))
                if(self.cfg.render.show):
                    env.render()
                if done or truncated:
                    break
        
        env.close()
        metric.plot_scores(save_fig_dir=env.cfg.output_dir)

        env.cfg.agent.action_type = old_action_type
        env.cfg.sim.t_max = old_t_max

        return metric.success_rates, metric.efficiency_scores, metric.effort_scores, metric.rewards, "SAM"


    
    def act(self, observation, **kwargs):
        raise NotImplementedError("The 'act' method is not implemented yet.")
        # parameters for planners
        model_eps = kwargs.get('model_eps', None)
        
        # load trained model for the first time
        if self.model is None:

            if model_eps is None:
                # self.model = PPO.load(os.path.join(self.model_path, self.model_name))
                pass
            else:
                model_checkpoint = self.model_name + '_' + model_eps + '_steps'
                # self.model = PPO.load(os.path.join(self.model_path, model_checkpoint))

        action, _ = self.model.predict(observation)
        return action
