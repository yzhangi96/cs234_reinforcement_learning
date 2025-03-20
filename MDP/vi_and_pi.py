### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (original_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = R[state, action] + T[state, action, :] @ (gamma * V)
    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)

    old_v = np.zeros(num_states)
    ep = 1
    while ep > tol:
        value_function = np.array([bellman_backup(state, policy[state], R, T, gamma, old_v) for state in range(num_states)])
        ep = max([abs(value_function[state]-old_v[state]) for state in range(num_states)])
        old_v = value_function.copy()

    return value_function


def policy_improvement(R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    for s in range(num_states):
        # all action pairs
        q_function = [bellman_backup(s, action, R, T, gamma, V_policy) for action in range(num_actions)]
        new_policy[s] = np.argmax(q_function)
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    l1_norm = 1
    old_policy = np.zeros(num_states, dtype=int)
    while l1_norm > 0:
        V_policy = policy_evaluation(policy, R, T, gamma, tol)
        policy = policy_improvement(policy, R, T, V_policy, gamma)
        l1_norm = np.linalg.norm(policy - old_policy, ord=1)
        old_policy = policy.copy()

    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    old_v = np.zeros(num_states)

    ep = 1
    while ep > tol:
        bellman_vals = np.array([[bellman_backup(state, action, R, T, gamma, old_v) for action in range(num_actions)] for state in range(num_states)])
        value_function = np.array([np.max(v) for v in bellman_vals])
        policy = np.array([np.argmax(v) for v in bellman_vals])
        ep = max([abs(value_function[state]-old_v[state]) for state in range(num_states)])
        old_v = value_function.copy()
    return value_function, policy


if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'MEDIUM'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.78
    
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])
