import time
import numpy as np
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_policy(env, policy, config_path="config.yaml"):
    cfg = load_config(config_path)
    episodes = cfg["episodes"]
    max_steps = cfg["max_steps"]

    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        total = 0
        for _ in range(max_steps):
            action = policy[state]
            state, reward, done, _ = env.step(action)
            total += reward
            if done:
                break
        total_rewards.append(total)
    return np.mean(total_rewards), total_rewards

import time

def time_algorithm(run_func):
    """
    Measures runtime of any function call.
    
    Parameters:
        run_func: A lambda or function with no arguments
    
    Returns:
        (result, duration) tuple
    """
    start_time = time.time()
    result = run_func()
    end_time = time.time()
    return result, end_time - start_time
