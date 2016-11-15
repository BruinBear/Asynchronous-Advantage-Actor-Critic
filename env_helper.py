import tensorflow as tf
import numpy as np
import gym
import pdb

def preprocess_atari_state(state, last_state=None):
	if last_state is None:
		return np.array(state)
	else:
		return np.subtract(state, last_state)

def main():
	env = gym.make('Breakout-v0')
	# test first state
	s0 = preprocess_atari_state(env.reset())
	# test state difference
	s1, _, _, _ = env.step(env.action_space.sample())
	s1 = preprocess_atari_state(s1, last_state=s0)

if __name__ == "__main__":
	main()