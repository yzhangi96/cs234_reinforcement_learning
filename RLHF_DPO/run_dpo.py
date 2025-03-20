import pathlib
from typing import Tuple
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from tqdm import trange

from data import load_data
from util import export_plot, np2torch, standard_error, eval_mode


LOGSTD_MIN = -10.0
LOGSTD_MAX = 2.0


class ActionSequenceModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        segment_len: int,
        lr: float = 1e-5,
    ):
        """
        Initialize an action sequence model.

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation space
        action_dim : int
            Dimension of the action space
        hidden_dim : int
            Number of neurons in the hidden layer
        segment_len : int
            Action segment length
        lr : float, optional
            Optimizer learning rate, by default 1e-3

        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.segment_len = segment_len

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * segment_len * action_dim),
        )
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the mean and standard deviation of the action distribution for each observation.

        Parameters
        ----------
        obs : torch.Tensor
            Batch of observations

        Returns
        -------
        Tuple[torch.Tensor]
            The means and standard deviations for the actions at future timesteps


        """
        if isinstance(obs, np.ndarray):
            obs = np2torch(obs)
        assert obs.ndim == 2
        batch_size = len(obs)
        net_out = self.net(obs)

        first_half, second_half = torch.split(net_out, self.segment_len * self.action_dim, dim=1)
        first_half = first_half.view(batch_size, self.segment_len, self.action_dim)
        second_half = second_half.view(batch_size, self.segment_len, self.action_dim)
        mean = torch.tanh(first_half)
        log_std_vector = torch.clamp(second_half, LOGSTD_MIN, LOGSTD_MAX)
        std = torch.exp(log_std_vector)

        return mean, std

    def distribution(self, obs: torch.Tensor) -> D.Distribution:
        """
        Take in a batch of observations and return a batch of action sequence distributions.

        Parameters
        ----------
        obs : torch.Tensor
            A tensor of observations

        Returns
        -------
        D.Distribution
            The action sequence distributions


        """
        mean, std = self.forward(obs)
        normal_dist = D.Normal(mean, std)
        return D.Independent(normal_dist, 2)


    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Return an action given an observation

        Parameters
        ----------
        obs : np.ndarray
            Single observation

        Returns
        -------
        np.ndarray
            The selected action

        """
        full_action, _ = self.forward(np2torch(obs).unsqueeze(0))
        return full_action[0, 0, :].detach().cpu().numpy()


class SFT(ActionSequenceModel):
    def update(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Pre-train a policy given an action sequence for an observation.

        Parameters
        ----------
        obs : torch.Tensor
            The start observation
        actions : torch.Tensor
            A plan of actions for the next timesteps

        """
        distr = self.distribution(obs)
        log_probs = distr.log_prob(actions)
        loss = -log_probs.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.net.parameters(), 1)
        self.optimizer.step()

        return loss.item()


class DPO(ActionSequenceModel):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        segment_len: int,
        beta: float,
        lr: float = 1e-5,
    ):
        super().__init__(obs_dim, action_dim, hidden_dim, segment_len, lr=lr)
        self.beta = beta

    def update(
        self,
        obs: torch.Tensor,
        actions_w: torch.Tensor,
        actions_l: torch.Tensor,
        ref_policy: nn.Module,
    ):
        """
        Run one DPO update step

        Parameters
        ----------
        obs : torch.Tensor
            The current observation
        actions_w : torch.Tensor
            The actions of the preferred trajectory
        actions_l : torch.Tensor
            The actions of the other trajectory
        ref_policy : nn.Module
            The reference policy

        """
        with torch.no_grad():
            ref_distr = ref_policy.distribution(obs)
            ref_log_probw = ref_distr.log_prob(actions_w)
            ref_log_probl = ref_distr.log_prob(actions_l)

        distr = self.distribution(obs)
        log_probw = distr.log_prob(actions_w)
        log_probl = distr.log_prob(actions_l)

        inner_obj = self.beta * ((log_probw - ref_log_probw) - (log_probl - ref_log_probl))
        log_sigmoid = torch.nn.functional.logsigmoid(inner_obj)
        loss = -log_sigmoid.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.net.parameters(), 1)
        self.optimizer.step()


        return loss.item()


def evaluate(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    np.random.seed(seed + 3)

    total_reward = 0
    for _ in range(env.spec.max_episode_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if done:
            break
    return total_reward


def get_batch(dataset, batch_size):
    obs1, obs2, act1, act2, label, act = dataset.sample(batch_size)
    obs = obs1[:, 0]
    assert torch.allclose(obs, obs2[:, 0])

    # Initialize assuming 1st actions preferred,
    # then swap where label = 1 (indicating 2nd actions preferred)
    actions_w = act1.clone()
    actions_l = act2.clone()
    swap_indices = label.nonzero()[:, 0]
    actions_w[swap_indices] = act2[swap_indices]
    actions_l[swap_indices] = act1[swap_indices]
    return obs, actions_w, actions_l, act


def main(args):
    output_path = pathlib.Path(__file__).parent.joinpath(
        args.output_path, f"Hopper-v4-{args.algo}-seed={args.seed}"
    )
    print(f"Output path: {output_path}")
    model_output = output_path.joinpath(f"{args.algo}.pt")
    scores_output = output_path.joinpath("scores.npy")
    plot_output = output_path.joinpath("scores.png")

    env = gym.make(args.env_name, terminate_when_unhealthy=False)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # DPO assumes preferences are strict, so we ignore the equally preferred pairs
    pref_data = load_data(args.dataset_path, strict_pref_only=True)
    segment_len = pref_data.sample(1)[0].size(1)
    all_returns = []

    print("Loading pretrained policy")
    sft = SFT(obs_dim, action_dim, args.hidden_dim, segment_len, lr=args.lr)
    sft.load_state_dict(torch.load("data/pretrain.pt", weights_only=True))

    print("Evaluating pretrained policy")
    with eval_mode(sft):
        returns = [evaluate(env, sft.act, seed=i + 1) for i in range(args.num_eval_episodes)]
    all_returns.append(np.mean(returns))
    print(f"Return: {np.mean(returns):.2f} +/- {standard_error(returns):.2f}")
    print("")

    # fix seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    best_model = None
    if args.algo == "sft":
        print("Training SFT")

        for step in trange(args.num_sft_steps):
            obs, actions_w, _, _ = get_batch(pref_data, args.batch_size)
            sft.update(obs, actions_w)

            if (step + 1) % args.eval_period == 0:
                print("Evaluating SFT policy")
                with eval_mode(sft):
                    returns = [
                        evaluate(env, sft.act, seed=i + 1) for i in range(args.num_eval_episodes)
                    ]
                print(f"Return: {np.mean(returns):.2f} +/- {standard_error(returns):.2f}")
                avg_return = np.mean(returns)
                if len(all_returns) == 0 or avg_return >= np.max(all_returns):
                    best_model = copy.deepcopy(sft)
                    print("New Best Checkpoint")
                all_returns.append(avg_return)
    else:
        print("Training DPO")
        dpo = DPO(obs_dim, action_dim, args.hidden_dim, segment_len, args.beta, lr=args.lr)
        dpo.net.load_state_dict(sft.net.state_dict())  # init with SFT parameters
        for step in trange(args.num_dpo_steps):
            obs, actions_w, actions_l, _ = get_batch(pref_data, args.batch_size)
            dpo.update(obs, actions_w, actions_l, sft)

            if (step + 1) % args.eval_period == 0:
                print("Evaluating DPO policy")
                with eval_mode(dpo):
                    returns = [
                        evaluate(env, dpo.act, seed=i + 1) for i in range(args.num_eval_episodes)
                    ]
                print(f"Return: {np.mean(returns):.2f} +/- {standard_error(returns):.2f}")
                avg_return = np.mean(returns)
                if len(all_returns) == 0 or avg_return >= np.max(all_returns):
                    best_model = copy.deepcopy(dpo)
                    print("New Best Checkpoint")
                all_returns.append(avg_return)

    # Log the results
    with open(model_output, "wb") as f:
        print(f"saving best_checkpoint to {model_output}")
        torch.save(best_model, f)

    np.save(scores_output, all_returns)
    export_plot(all_returns, "Returns", "Hopper-v4", plot_output)


if __name__ == "__main__":
    from argparse import ArgumentParser

    default_dataset = pathlib.Path(__file__).parent.joinpath("data", "prefs-hopper.npz")
    parser = ArgumentParser()
    parser.add_argument("--env-name", default="Hopper-v4")
    parser.add_argument("--dataset-path", default=default_dataset)
    parser.add_argument("--output-path", default="results_dpo")
    parser.add_argument("--algo", type=str, default="dpo", help="dpo/sft")
    # common parameters
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-period", type=int, default=2000)
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    # sft parameters
    parser.add_argument("--num-sft-steps", type=int, default=20000)
    # DPO parameters
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-dpo-steps", type=int, default=20000)

    main(parser.parse_args())
