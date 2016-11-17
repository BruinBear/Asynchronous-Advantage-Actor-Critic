import tensorflow as tf
import gym
import pdb
import numpy as np
from collections import namedtuple
import threading
import time
import env_helper
import random

from actor_critic_network import ActorCriticNetwork

SARS = namedtuple('SARS', ['state', 'action', 'reward', 'next_state'], verbose=True)

class Trainer(object):
	def __init__(self, id, env, sess, global_t_lock, ac_network):
		self.id = id
		self.env = env
		self.sess = sess
		self.state = env.reset()
		self.local_t = 0
		self.ac_network = ac_network

	def run(self, coord):
		self.run_n_steps(3)

	def run_n_steps(self, n):
		global global_t
		print('thread {} run n steps'.format(self.id))
		time.sleep(random.random())
		history = []

		for i in range(n):
			action = self.ac_network.pick_action(self.sess, self.state)
			next_state, reward, done, info = self.env.step(action)
			# add to history
			history.append(SARS(self.state, action, reward, next_state))
			# increment counters
			with global_t_lock:
				global_t += 1
				print(global_t)
			self.local_t += 1
			# update state
			self.state = next_state

			if done:
				break
		return history


global_t = 0
global_t_lock = threading.Lock()

def main():
	trainers = []
	env = gym.make('Breakout-v0')
	ac_network = ActorCriticNetwork(env.action_space.n)

	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		sess.run(tf.initialize_all_variables())

		env = gym.make('Breakout-v0')
		# Create 10 threads that run 'MyLoop()'
		for t in range(3):
			trainers.append(Trainer(t, env, sess, global_t_lock, ac_network))
		threads = [threading.Thread(target=trainer.run, args=(coord,)) for trainer in trainers]

		# Start the threads and wait for all of them to stop.
		for t in threads: t.start()
		coord.join(threads)


if __name__ == "__main__":
	main()
