import numpy as np
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def value_iteration(env, config_path="config.yaml"):
    cfg = load_config(config_path)
    gamma = float(cfg["gamma"])
    theta = float(cfg["theta"])  # Convergence threshold
    print(f"[DP] gamma={gamma}, theta={theta} (type: {type(theta)})")

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    def one_step_lookahead(state, V):
        A = np.zeros(n_actions)
        for a in range(n_actions):
            env.agent_pos = env.get_state_coords(state)
            next_state, reward, done, _ = env.step(a)
            A[a] = reward + gamma * V[next_state] * (not done)
        return A

    iterations = 0
    while True:
        delta = 0
        for s in env.get_valid_states():
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        iterations += 1
        if delta < theta:
            break

    for s in env.get_valid_states():
        A = one_step_lookahead(s, V)
        policy[s] = np.argmax(A)

    return policy, V, iterations  # <-- return iterations too
