import numpy as np
import random
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_episode(env, policy, epsilon):
    episode = []
    state = env.reset()

    done = False
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = policy[state]
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return episode


def train_monte_carlo(env, config_path="config.yaml"):
    cfg = load_config(config_path)
    gamma = cfg["gamma"]
    alpha = cfg["alpha"]
    epsilon = cfg["epsilon"]
    episodes = cfg["episodes"]
    max_steps = cfg["max_steps"]

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    policy = np.zeros(n_states, dtype=int)
    reward_per_episode = []

    convergence_episode = None
    unchanged_count = 0
    prev_policy = policy.copy()

    for ep in range(episodes):
        episode = generate_episode(env, policy, epsilon)
        G = 0
        visited = set()
        total_reward = 0

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G
            total_reward += r
            if (s, a) not in visited:
                visited.add((s, a))
                Q[s][a] += alpha * (G - Q[s][a])
        
        # Update policy after full episode
        for s in env.get_valid_states():
            policy[s] = np.argmax(Q[s])

        # Track convergence
        if np.array_equal(policy, prev_policy):
            unchanged_count += 1
            if unchanged_count == 10 and convergence_episode is None:
                convergence_episode = ep
        else:
            unchanged_count = 0
            prev_policy = policy.copy()

        reward_per_episode.append(total_reward)

    V = np.max(Q, axis=1)
    if convergence_episode is None:
        convergence_episode = episodes

    return policy, V, Q, reward_per_episode, convergence_episode
