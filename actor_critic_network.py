import tensorflow as tf
import gym
import env_helper
import pdb
import numpy as np

class ActorCriticNetwork:

	def __init__(self, action_size):
		self.action_size = action_size

		# declaring placeholders
		self.input_state = tf.placeholder(tf.uint8, shape=[None,210,160,3])
		self.value_target = tf.placeholder(tf.float32, shape=[None])
		self.action_target = tf.placeholder(tf.uint8, shape=[None]) # need to convert to one hot
		self.action_target = tf.one_hot(self.action_target, action_size)
		self.action_advantage = tf.placeholder(tf.float32, shape=[None,1])

		# preprocess
		self.processed_input = self.build_shared_net(self.input_state)

		# policy
		self.action_prob_prediction, self.action_loss = self.build_policy_net()

		# value
		self.value_prediction, self.value_loss = self.build_value_net()

		# combined loss
		self.loss = tf.add_n([self.action_loss, self.value_loss])

		# optimizer
		self.optimizer = tf.train.RMSPropOptimizer(0.001, 0.99, 0.0, 1e-6)
		
		# train op
		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
		self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
			global_step=tf.contrib.framework.get_global_step())

	def build_shared_net(self, input):
		with tf.variable_scope('shared_preprocess'):
			hidden = tf.to_float(input)/255.0
			hidden = tf.contrib.layers.convolution2d(hidden, num_outputs=32, kernel_size=5, stride=2, activation_fn=tf.nn.relu)
			hidden = tf.contrib.layers.max_pool2d(hidden, kernel_size=2, stride=2, padding='VALID')
			hidden = tf.contrib.layers.convolution2d(hidden, num_outputs=32, kernel_size=5, stride=2, activation_fn=tf.nn.relu)
			hidden = tf.contrib.layers.max_pool2d(hidden, kernel_size=2, stride=2, padding='VALID')
			hidden = tf.contrib.layers.convolution2d(hidden, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
			hidden = tf.contrib.layers.max_pool2d(hidden, kernel_size=2, stride=2, padding='VALID')
			hidden = tf.contrib.layers.convolution2d(hidden, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
			hidden = tf.contrib.layers.flatten(hidden)
			out = tf.contrib.layers.fully_connected(hidden, num_outputs=512)
			return out

	def build_policy_net(self):
		with tf.variable_scope('policy_net'):
			action_logits = tf.contrib.layers.fully_connected(self.processed_input, self.action_size, activation_fn=None)
			action_prob_prediction = tf.nn.softmax(action_logits) + 1e-8

			log_action_prob_prediction = tf.log(action_prob_prediction)
			log_action_prob_target = tf.reduce_sum(log_action_prob_prediction * self.action_target, 1)
			policy_gain = tf.reduce_sum(log_action_prob_target * self.action_advantage, name='policy_loss')

			entropy = -tf.reduce_sum(action_prob_prediction * log_action_prob_prediction, 1)
			entropy_gain = tf.reduce_sum(entropy)

			loss =  -(policy_gain + 0.01 * entropy_gain)

			return action_prob_prediction, loss

	def build_value_net(self):
		with tf.variable_scope('value_net'):
			value_prediction = tf.contrib.layers.fully_connected(self.processed_input, 1, activation_fn=None)
			loss = -tf.reduce_sum(tf.squared_difference(value_prediction, self.value_target))

			return value_prediction, loss

	def pick_action(self, sess, state):
		prob = sess.run(self.action_prob_prediction, feed_dict={self.input_state: [state]})[0]
		action = np.random.choice(range(self.action_size), 1, p = prob)
		return action[0]

def main():
	env = gym.make('Breakout-v0')
	# test first state
	s0 = env_helper.preprocess_atari_state(env.reset())

	with tf.Session() as sess:
		with tf.device("/cpu:0"):
			ac_network = ActorCriticNetwork(env.action_space.n)
			sess.run(tf.initialize_all_variables())
			# random behavior
			for i in range(100):
				env.render()
				a = ac_network.pick_action(sess, s0)
				env.step(a)

if __name__ == "__main__":
	main()
