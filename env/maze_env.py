import numpy as np
import gym
from gym import spaces

class MazeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid = np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 2],  # 2 = goal
        ])
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos
        self.goal_pos = (3, 6)
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Discrete(np.prod(self.grid.shape))

    def reset(self):
        self.agent_pos = self.start_pos
        return self._get_state_index()

    def step(self, action):
        y, x = self.agent_pos
        if action == 0 and y > 0: y -= 1  # up
        elif action == 1 and y < self.grid.shape[0] - 1: y += 1  # down
        elif action == 2 and x > 0: x -= 1  # left
        elif action == 3 and x < self.grid.shape[1] - 1: x += 1  # right

        if self.grid[y, x] == 1:  # wall
            y, x = self.agent_pos

        self.agent_pos = (y, x)
        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else -0.01
        return self._get_state_index(), reward, done, {}

    def _get_state_index(self):
        y, x = self.agent_pos
        return y * self.grid.shape[1] + x

    def get_state_coords(self, index):
        return divmod(index, self.grid.shape[1])

    def get_valid_states(self):
        return [y * self.grid.shape[1] + x
                for y in range(self.grid.shape[0])
                for x in range(self.grid.shape[1])
                if self.grid[y, x] != 1]
