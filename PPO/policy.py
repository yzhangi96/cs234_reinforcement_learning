import torch
import torch.nn as nn
import torch.distributions as ptd

from network_utils import np2torch, device


class BasePolicy:
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution

        Returns an object representing the policy's conditional
        distribution(s) given the observations.
        """
        raise NotImplementedError

    def act(self, observations, return_log_prob = False):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)

        """
        observations = np2torch(observations)
        distribution = self.action_distribution(observations)
        sampled_actions = distribution.sample()
        log_probs = distribution.log_prob(sampled_actions).detach().numpy()
        sampled_actions = sampled_actions.numpy()
        if return_log_prob:
            return sampled_actions, log_probs
        return sampled_actions


class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network

        """
        distribution = ptd.Categorical(logits=self.network(observations))
        return distribution


class GaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, network, action_dim):

        nn.Module.__init__(self)
        self.network = network
        self.log_std = nn.Parameter(torch.zeros(action_dim))


    def std(self):
        """
        Returns:
            std: torch.Tensor of shape [dim(action space)]

        The return value contains the standard deviations for each dimension
        of the policy's actions.
        """

        std = torch.exp(self.log_std)
        return std

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: an instance of a subclass of
                torch.distributions.Distribution representing a diagonal
                Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.std()

        """

        loc = self.network(observations)
        var = torch.square(self.std())
        distribution = ptd.MultivariateNormal(loc, torch.diag(var))

        return distribution
