from termcolor import cprint
import numpy as np


class BitFlipEnv():
    """
    A simple bit flip environment
    Bit of the current state flips as an action
    Reward of -1 for each step
    """
    def __init__(self):
        self.n_bits = None
        self.state = None
        self.goal = None
        self.max_episode_steps = None

    def _make(self, n_bits, max_episode_steps=None):
        self.n_bits = n_bits
        self.state = np.random.randint(2, size=self.n_bits)
        self.goal = np.random.randint(2, size=self.n_bits)
        self.max_episode_steps = self.n_bits if max_episode_steps == None else max_episode_steps

        self.n_observation = n_bits
        self.n_goal = n_bits
        self.n_action =1

    def _reset(self):
        """
        Resets the environment with new state and goal
        """
        self.state = np.random.randint(2, size=self.n_bits)
        self.goal = np.random.randint(2, size=self.n_bits)
        self.step = 0

        return self.state, self.goal

    def _step(self, action):
        """
        Returns updated_state, reward, and done for the step taken
        """
        self.state[action] = self.state[action] ^ 1
        done = False
        trunc = False
        # breakpoint()
        if np.array_equal(self.state, self.goal):
            done = True
            reward = 0
        else:
            reward = -1

        if (self.step+1) == self.max_episode_steps:
            trunc = True
        else:
            trunc = False
            self.step += 1
            
        return np.copy(self.state), reward, done, trunc

    def _render(self):
        """
        Prints the current state
        """
        cprint(f'Step{self.step} / Current State: {np.array2string(self.state)}', "green")