#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:10:26 2018

@author: philippeamadei
"""
import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

gym.envs.register(
    id='CartPoleLong-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

env = gym.make('CartPoleLong-v0')

env = gym.wrappers.Monitor(env, '/Users/philippeamadei/Desktop/MachineLearning/ReinforcementLearning/PoleCart/VideoRecord/', video_callable=lambda episode_id: episode_id%100==0)

obs = env.reset()

n_inputs = 4
n_hidden = 4
n_outputs = 1
initializer = tf.contrib.layers.variance_scaling_initializer()
learning_rate = 0.01
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu,
                         weights_initializer=initializer) 
logits = fully_connected(hidden, n_outputs, activation_fn=None,
                         weights_initializer=initializer) 
outputs = tf.nn.sigmoid(logits)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs]) 
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits) 
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy) 
gradients = [grad for grad, variable in grads_and_vars] 
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape()) 
    gradient_placeholders.append(gradient_placeholder) 
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed) 
init = tf.global_variables_initializer()
saver = tf.train.Saver()


def discount_rewards(rewards, discount_rate): 
    discounted_rewards = np.empty(len(rewards)) 
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards 
    return discounted_rewards
def discount_and_normalize_rewards(all_rewards, discount_rate): 
    all_discounted_rewards = [discount_rewards(rewards,discount_rate)
                                for rewards in all_rewards] 
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std
            for discounted_rewards in all_discounted_rewards]
    

n_iterations = 250 
n_max_steps = 1000 
n_games_per_update = 10
save_iterations = 10 
discount_rate = 0.95

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        all_rewards = [] # all sequences of raw rewards for each episode all_gradients = [] # gradients saved at each step of each episode for game in range(n_games_per_update):
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = [] # all raw rewards from the current episode current_gradients = [] # all gradients from the current episode obs = env.reset()
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run( [action, gradients],
                                                     feed_dict={X: obs.reshape(1, n_inputs)}) # one obs 
                obs, reward, done, info = env.step(action_val[0][0]) 
                current_rewards.append(reward) 
                current_gradients.append(gradients_val)
                if done:
                    print(str(step))
                    break
                all_rewards.append(current_rewards) 
                all_gradients.append(current_gradients)
        # At this point we have run the policy for 10 episodes, and we are # ready for a policy update using the algorithm described earlier. 
        all_rewards = discount_and_normalize_rewards(all_rewards,discount_rate) 
        feed_dict = {}
        for var_index, grad_placeholder in enumerate(gradient_placeholders):
        # multiply the gradients by the action scores, and compute the mean 
            mean_gradients = np.mean(
                    [reward * all_gradients[game_index][step][var_index] 
                    for game_index, rewards in enumerate(all_rewards) 
                    for step, reward in enumerate(rewards)],
                    axis=0)
            feed_dict[grad_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict) 
        if iteration % save_iterations == 0:
            saver.save(sess, "/Users/philippeamadei/Desktop/MachineLearning/ReinforcementLearning/PoleCart/Checkpoint/CartPoleSave.ckpt")