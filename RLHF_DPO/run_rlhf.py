import pathlib
from typing import Tuple

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from data import load_data
from util import export_plot, np2torch


class RewardModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, r_min: float, r_max: float):
        """Initialize a reward model

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation space
        action_dim : int
            Dimension of the action space
        hidden_dim : int
            Number of neurons in the hidden layer
        r_min : float
            Minimum reward value
        r_max : float
            Maximum reward value

        """

        super().__init__()
        input_size = obs_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.AdamW(self.net.parameters())
        self.r_min = r_min
        self.r_max = r_max

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward callback for the RewardModel

        Parameters
        ----------
        obs : torch.Tensor
            Batch of observations
        action : torch.Tensor
            Batch of actions

        Returns
        -------
        torch.Tensor
            Batch of predicted rewards

        """
        if obs.ndim == 3:
            B, T = obs.shape[:2]
            assert action.ndim == 3 and action.shape[:2] == (B, T)
            obs = obs.reshape(-1, obs.shape[-1])
            action = action.reshape(-1, action.shape[-1])
            needs_reshape = True
        else:
            needs_reshape = False

        input = torch.cat((obs, action), dim=1)
        pred = self.net(input).squeeze(dim=1)
        rewards = self.r_min + (self.r_max - self.r_min) * pred

        if needs_reshape:
            rewards = rewards.reshape(B, T)
        return rewards

    def compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Given an (observation, action) pair, return the predicted reward.

        Parameters
        ----------
        obs : (obs_dim, )
            A numpy array with an observation.
        action : np.ndarray (act_dim, )
            A numpy array with an action

        Returns
        -------
        float
            The predicted reward for the state-action pair.
        """
        obs_torch = np2torch(obs)
        action_torch = np2torch(action)

        # batch dimension
        pred_reward = self.forward(torch.unsqueeze(obs_torch, 0),
                              torch.unsqueeze(action_torch, 0))
        return pred_reward.item()

    def update(self, batch: Tuple[torch.Tensor]):
        """Given a batch of data, update the reward model.

        Parameters
        ----------
        batch : Tuple[torch.Tensor]
            A batch with two trajectories (observations and actions) and a label
            encoding preferences (0 if the first one is preferred, 1 if the second
            one is preferred, and 0.5 if they are equally preferred).

        """
        obs1, obs2, act1, act2, label, _ = batch

        pred_reward1 = self.forward(obs1, act1)
        pred_reward2 = self.forward(obs2, act2)

        sum_exp1 = torch.sum(pred_reward1, dim=1)
        sum_exp2 = torch.sum(pred_reward2, dim=1)

        input = torch.stack((sum_exp1, sum_exp2), axis=1)
        target = torch.cat((1-label.unsqueeze(1), label.unsqueeze(1)), axis=1)
        loss = F.cross_entropy(input, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self._obs = obs.copy()
        return obs, info

    def step(self, action):
        next_obs, og_reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward_fn.compute_reward(self._obs, action)
        info["og_reward"] = og_reward
        self._obs = next_obs.copy()
        return next_obs, reward, terminated, truncated, info


def evaluate(env, policy):
    model_return, og_return = 0, 0
    T = env.spec.max_episode_steps
    obs, _ = env.reset()
    for _ in range(T):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        model_return += reward
        og_return += info["og_reward"]
        if done:
            break
    return model_return, og_return


class EvalCallback(sb3.common.callbacks.BaseCallback):
    def __init__(self, eval_period, num_episodes, env, policy):
        super().__init__()
        self.eval_period = eval_period
        self.num_episodes = num_episodes
        self.env = env
        self.policy = policy

        self.original_returns = []
        self.learned_returns = []

    def _on_step(self):
        if self.n_calls % self.eval_period == 0:
            print(f"Evaluating after {self.n_calls} steps")
            model_returns, og_returns = [], []
            for _ in range(self.num_episodes):
                model_return, og_return = evaluate(self.env, self.policy)
                model_returns.append(model_return)
                og_returns.append(og_return)
            print(f"Model return: {np.mean(model_returns):.2f} +/- {np.std(model_returns):.2f}")
            print(f"Original return: {np.mean(og_returns):.2f} +/- {np.std(og_returns):.2f}")
            self.original_returns.append(np.mean(og_returns))
            self.learned_returns.append(np.mean(model_returns))

        # If the callback returns False, training is aborted early.
        return True


def main(args):
    output_path = pathlib.Path(__file__).parent.joinpath(
        "results_rlhf",
        f"Hopper-v4-rlhf-seed={args.seed}",
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_output = output_path.joinpath("model.zip")
    original_scores_output = output_path.joinpath("original_scores.npy")
    learned_scores_output = output_path.joinpath("learned_scores.npy")
    original_plot_output = output_path.joinpath("original_scores.png")
    learned_plot_output = output_path.joinpath("learned_scores.png")

    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    data = load_data(args.dataset_path)
    reward_model = RewardModel(
        obs_dim,
        action_dim,
        args.reward_model_hidden_dim,
        args.reward_min,
        args.reward_max,
    )

    print("Training reward model")
    for _ in trange(args.reward_model_steps):
        reward_model.update(data.sample(args.reward_model_batch_size))

    custom_reward_env = CustomRewardEnv(env, reward_model)
    agent = sb3.PPO("MlpPolicy", custom_reward_env, verbose=1)
    eval_callback = EvalCallback(
        args.eval_period,
        args.num_eval_episodes,
        custom_reward_env,
        lambda obs: agent.predict(obs)[0],
    )
    agent.learn(args.rl_steps, callback=eval_callback)

    # Log the results
    if not output_path.exists():
        output_path.mkdir(parents=True)
    agent.save(model_output)
    np.save(original_scores_output, eval_callback.original_returns)
    np.save(learned_scores_output, eval_callback.learned_returns)

    export_plot(
        eval_callback.original_returns,
        "Original returns",
        "Hopper-v4",
        original_plot_output,
    )
    export_plot(
        eval_callback.learned_returns,
        "Learned returns",
        "Hopper-v4",
        learned_plot_output,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--env-name", default="Hopper-v4")
    parser.add_argument(
        "--dataset-path",
        default=pathlib.Path(__file__).parent.joinpath("data", "prefs-hopper.npz"),
    )
    parser.add_argument("--reward-min", type=float, default=0.0)
    parser.add_argument("--reward-max", type=float, default=1.0)
    parser.add_argument("--reward-model-hidden-dim", type=int, default=64)
    parser.add_argument("--reward-model-steps", type=int, default=100000)
    parser.add_argument("--reward-model-batch-size", type=int, default=64)
    parser.add_argument("--rl-steps", type=int, default=1000000)
    parser.add_argument("--eval-period", type=int, default=1000)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
