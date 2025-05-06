import numpy as np
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_td0(env, config_path="config.yaml"):
    cfg = load_config(config_path)
    gamma = cfg["gamma"]
    alpha = cfg["alpha"]
    episodes = cfg["episodes"]
    max_steps = cfg["max_steps"]

    n_states = env.observation_space.n
    V = np.zeros(n_states)
    reward_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = env.action_space.sample()  # random policy
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # TD(0) update
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state

            if done:
                break

        reward_per_episode.append(total_reward)

    return V, reward_per_episode


def extract_policy(env, V, gamma):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.zeros(n_states, dtype=int)

    for s in env.get_valid_states():
        best_value = -np.inf
        best_action = 0
        for a in range(n_actions):
            env.agent_pos = env.get_state_coords(s)
            next_state, reward, done, _ = env.step(a)
            value = reward + gamma * V[next_state] * (not done)
            if value > best_value:
                best_value = value
                best_action = a
        policy[s] = best_action
    return policy
