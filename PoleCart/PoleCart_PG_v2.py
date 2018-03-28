#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 22:29:02 2018

@author: philippeamadei
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import gym
import matplotlib.pyplot as plt

class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=None,
        save_path=None
    ):

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay
        
        self.save_path = None
        if save_path is not None:
            self.save_path = save_path
            

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        self.build_network()

        self.sess = tf.Session()

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        #tf.summary.FileWriter("logs/", self.sess.graph)
        
        # 'Saver' op to save and restore all the variables            

        self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()
        
        # Restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)
            
    def store_transition(self, s, a, r):
        """
            Store play memory for training
            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_observations.append(s)
        self.episode_rewards.append(r)
        self.episode_actions.append(a)


    def choose_action(self, observation):
        """
            Choose action based on observation
            Arguments:
                observation: array of state, has shape (num_features)
            Returns: index of action we want to choose
        """
        # Reshape observation to (1, num_features)
        observation = observation[np.newaxis, :]

        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    def learn(self):
        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # Train on episode
        self.sess.run(self.train_op, feed_dict={
             self.X: np.vstack(self.episode_observations), # shape [ examples, number of inputs]
             self.Y: np.array(self.episode_actions), # shape [actions, ]
             self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []
        
        # Save checkpoint
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            
        return discounted_episode_rewards_norm

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards



    def build_network(self):
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, [None, self.n_x], name="X")
            self.Y = tf.placeholder(tf.int32, [None, ], name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        A1 = tf.layers.dense(
            inputs=self.X,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        A2 = tf.layers.dense(
            inputs=A1,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        # fc3
        Z3 = tf.layers.dense(
            inputs=A2,
            units=self.n_y,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc3'
        )

        # Softmax outputs
        self.outputs_softmax = tf.nn.softmax(Z3, name='A3')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Z3, labels=self.Y)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            

gym.envs.register(
    id='CartPoleLong-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=350,
    reward_threshold=195.0,
)

env = gym.make('CartPoleLong-v0')
#env = gym.wrappers.Monitor(env, '/Users/philippeamadei/Desktop/MachineLearning/ReinforcementLearning/PoleCart/VideoRecord/', video_callable=lambda episode_id: episode_id%100==0)

# Policy gradient has high variance, seed for reproducability
env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)


RENDER_ENV = False
EPISODES = 200
rewards = []
RENDER_REWARD_MIN = 50

if __name__ == "__main__":


    # Load checkpoint
    load_path = '/Users/philippeamadei/Desktop/MachineLearning/ReinforcementLearning/PoleCart/Checkpoint/model.ckpt' #"output/weights/CartPole-v0.ckpt"
    save_path = '/Users/philippeamadei/Desktop/MachineLearning/ReinforcementLearning/PoleCart/Checkpoint/model.ckpt' #"output/weights/CartPole-v0-temp.ckpt"

    PG = PolicyGradient(
        n_x = env.observation_space.shape[0],
        n_y = env.action_space.n,
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=load_path,
        save_path=save_path
    )

    learningCurve = []
    for episode in range(EPISODES):

        observation = env.reset()
        episode_reward = 0

        while True:
            #if RENDER_ENV: env.render()

            # 1. Choose an action based on observation
            action = PG.choose_action(observation)

            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)

            # 3. Store transition for training
            PG.store_transition(observation, action, reward)

            if done:
                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Max reward so far: ", max_reward_so_far)
                learningCurve += [episode_rewards_sum]

                # 4. Train neural network
                discounted_episode_rewards_norm = PG.learn()

                # Render env if we get to rewards minimum
                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True
                break

            # Save new observation
            observation = observation_
    import pandas as pd
    pd.Series(learningCurve).plot()
            