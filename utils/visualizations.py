import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np



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
