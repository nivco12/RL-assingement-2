import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd



def animate_agent_path(env, path, title="Agent Path"):
    plt.close()  # close previous plots

    # cmap = ListedColormap([
    #     '#1f77b4',  # 0 = path (blue)
    #     '#2ca02c',  # 1 = wall (green)
    #     '#ffdf00',  # 2 = goal (yellow)
    #     '#d62728'   # 9 = agent (red)
    # ])
    cmap = ListedColormap([
    '#1f77b4',  # 0 = path (blue)
    '#2ca02c',  # 1 = wall (green)
    '#ffdf00',  # 2 = goal (yellow)
    '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4',
    '#d62728'   # 9 = agent (red)
    ])


    frames = []
    for state in path:
        grid = env.grid.copy()
        y, x = env.get_state_coords(state)
        grid[y, x] = 9
        frames.append(grid)

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=9)
    plt.axis('off')

    def update(frame):
        im.set_array(frame)
        return [im]

    #ani = animation.FuncAnimation(fig, update, frames=frames, interval=300, blit=True)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=300, blit=False)

    plt.title(title)
    plt.show()



def plot_value_map(env, V, title="State Value Function"):
    grid = np.full_like(env.grid, np.nan, dtype=float)
    for s in env.get_valid_states():
        y, x = env.get_state_coords(s)
        grid[y, x] = V[s]

    plt.figure(figsize=(6,4))
    plt.imshow(grid, cmap="viridis", interpolation='none')
    plt.colorbar(label='V(s)')
    plt.title(title)
    plt.axis('off')
    plt.show()




def show_runtime_table(runtime_dp, runtime_mc, runtime_td0, runtime_q, runtime_sarsa, runtimes_td_lambda):
    """
    Displays a sorted runtime comparison table for all RL algorithms.
    
    Parameters:
        runtime_dp (float): Runtime of Dynamic Programming
        runtime_mc (float): Runtime of Monte Carlo
        runtime_td0 (float): Runtime of TD(0)
        runtime_q (float): Runtime of Q-learning
        runtime_sarsa (float): Runtime of SARSA
        runtimes_td_lambda (list of float): List of runtimes for TD(λ) with λ = [0.2, 0.5, 0.8]
    """
    
    algo_names = [
        "DP",
        "Monte Carlo",
        "TD(0)",
        "Q-learning",
        "SARSA",
        "TD(λ=0.2)",
        "TD(λ=0.5)",
        "TD(λ=0.8)"
    ]

    algo_runtimes = [
        runtime_dp,
        runtime_mc,
        runtime_td0,
        runtime_q,
        runtime_sarsa,
        runtimes_td_lambda[0],
        runtimes_td_lambda[1],
        runtimes_td_lambda[2]
    ]

    timing_df = pd.DataFrame({
        "Algorithm": algo_names,
        "Runtime (seconds)": algo_runtimes
    })

    timing_df = timing_df.sort_values("Runtime (seconds)", ascending=True).reset_index(drop=True)

    print("### Runtimes of the Algorithms (from Fastest to Slowest):")
    return timing_df





def show_avg_reward_table(avg_reward_dp, avg_reward_mc, avg_reward_td0, avg_reward_q, avg_reward_sarsa, avg_rewards_td_lambda):
    """
    Displays a sorted runtime comparison table for all RL algorithms.
    
    """
    
    algo_names = [
        "DP",
        "Monte Carlo",
        "TD(0)",
        "Q-learning",
        "SARSA",
        "TD(λ=0.2)",
        "TD(λ=0.5)",
        "TD(λ=0.8)"
    ]

    algo_avg_rewards = [
        avg_reward_dp,
        avg_reward_mc,
        avg_reward_td0,
        avg_reward_q,
        avg_reward_sarsa,
        avg_rewards_td_lambda[0],
        avg_rewards_td_lambda[1],
        avg_rewards_td_lambda[2]
    ]

    rewards_df = pd.DataFrame({
        "Algorithm": algo_names,
        "Runtime (seconds)": algo_avg_rewards
    })

    rewards_df = rewards_df.sort_values("Runtime (seconds)", ascending=True).reset_index(drop=True)

    print("### Average rewards of the Algorithms ")
    return rewards_df
