import numpy as np
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_q_learning(env, config_path="config.yaml"):
    cfg = load_config(config_path)
    gamma = cfg["gamma"]
    alpha = cfg["alpha"]
    epsilon = cfg["epsilon"]
    episodes = cfg["episodes"]
    max_steps = cfg["max_steps"]

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    reward_per_episode = []
    convergence_episode = None
    prev_policy = np.zeros(n_states, dtype=int)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            if done:
                break

        reward_per_episode.append(total_reward)

        # Check for convergence
        current_policy = np.argmax(Q, axis=1)
        if np.array_equal(current_policy, prev_policy) and convergence_episode is None:
            convergence_episode = ep
        else:
            prev_policy = current_policy.copy()

    V = np.max(Q, axis=1)
    if convergence_episode is None:
        convergence_episode = episodes

    policy = np.argmax(Q, axis=1)
    return policy, V, Q, reward_per_episode, convergence_episode
