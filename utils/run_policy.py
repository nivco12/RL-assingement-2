def run_policy(env, policy, max_steps=100):
    state = env.reset()
    path = [state]
    for _ in range(max_steps):
        action = policy[state]
        state, reward, done, _ = env.step(action)
        path.append(state)
        if done:
            break
    return path
