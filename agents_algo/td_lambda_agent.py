import numpy as np
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_td_lambda(env, config_path="config_td_lambda.yaml"):
    cfg = load_config(config_path)
    gamma = cfg["gamma"]
    alpha = cfg["alpha"]
    lambdas = cfg["lambda"]
    episodes = cfg["episodes"]
    max_steps = cfg["max_steps"]

    if isinstance(lambdas, float) or isinstance(lambdas, int):
        lambdas = [lambdas]

    all_results = []

    for lambd in lambdas:
        print(f"Training TD(λ) with λ = {lambd}")
        n_states = env.observation_space.n
        V = np.zeros(n_states)
        reward_per_episode = []
        convergence_episode = None
        prev_V = V.copy()

        for ep in range(episodes):
            state = env.reset()
            E = np.zeros(n_states)  # Eligibility trace
            total_reward = 0

            for _ in range(max_steps):
                action = env.action_space.sample()  # On-policy (random)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                delta = reward + gamma * V[next_state] - V[state]
                E[state] += 1

                # Update all states
                V += alpha * delta * E
                E *= gamma * lambd

                state = next_state
                if done:
                    break

            reward_per_episode.append(total_reward)

            if np.allclose(V, prev_V, atol=1e-4) and convergence_episode is None:
                convergence_episode = ep
            prev_V = V.copy()

        if convergence_episode is None:
            convergence_episode = episodes

        policy = extract_policy(env, V, gamma)
        all_results.append({
            "lambda": lambd,
            "policy": policy,
            "value_table": V.copy(),
            "rewards": reward_per_episode,
            "convergence_ep": convergence_episode
        })

    return all_results

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
