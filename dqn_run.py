import numpy as np
import tensorflow as tf
import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import logz
import os
import time
import tensorflow.contrib.layers as layers

from atari_wrappers import *

# Neural net model for Deep Q learning  
def dqn_model (input, action_size, scope):
    with tf.variable_scope(scope, False):
        with tf.variable_scope("convnet"):
            output = layers.convolution2d(input, 32, 8, 4, tf.nn.relu) #output size 32, kernel size 8, stride 4
            output = layers.convolution2d(output, 64, 4, 2, tf.nn.relu)
            output = layers.convolution2d(output, 64, 3, 1, tf.nn.relu)
        
        output = layers.flatten(output)
        with tf.variable_scope("action_value"):
            output = layers.fully_connected(output, 512, tf.nn.relu)
            output = layers.fully_connected(output, action_size, None)
        
        return output



# Train Deep Q neural net
def dqn_learn (env, qFunction, session):
   img_h, img_w, img_c = env.observation_space.shape
   iShape = (img_h, img_w, 4 * img_c) #4 frames in sequence, this will give a sense of motion
   actionNum = env.action_space.n

   #observation, action, reward place holder
   obs = tf.placeholder(tf.float32, [None] + list (iShape)) #current
   act = tf.placeholder(tf.float32, [None])
   rew = tf.placeholder(tf.float32, [None])
   nObs = tf.placeholder(tf.float32, [None] + lost(iShape)) #next

   #if the next state is the end of an episode, no Q-value at the next state which means only current state reward contributes to the target
   done = tf.placeholder(tf.int32, [None])

   qf = qFunction(obs, actionNum, scope="qFunction")
   #we need a separate NN for target Q function
   targetQf = qFuction(nObs, actionNum, scope="tQFuction")
   #calculate q value for target Q fuction
   qTVal = rew + (1 - done) * 0.99 * tf.reduce_max(targetQf, axis=1)
   #current q value
   qVal = tf.reduce_sum(qf * tf.one_hot(act, actionNum), axis =1)
   #loss function should be Bellman error
   bellmanError = tf.reduce_mean(tf.square(qTval - qVal))
   #we need to collect Q fuction and target Q function variables to update target Q function periodically
   varQf = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gFunction")
   varTQf = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetQFunction')

   gradients = optimizer.compute_gradients(bellmanError, varQf)
   for i, (grad, var) in enumerate(gradients):
       if grad is not None:
           gradients[i] = (tf.clip_by_norm(grad, 10), var)
           
    dqnTrain = optimizer.apply_gradients(gradients)

    #Update target function periodically
    targetQVarsToUpdate = []
    for var, targetVar in zip(sorted(varQf, key=lambda v: v.name),
                              sorted(varTQf, key=lambda v: v.name)):
        targetQVarsToUpdate.append(varTQf.assign(var)) #this is basically copy Q function to target Q
    
    #Group the list of the assign operations
    targetQVarsToUpdate = tf.group(*targetQVarsToUpdate)

    replbuffer = replayBuffer(replay_buffer_size, 4) #four previous frames

    for t in trainingIter.count():
        #save the last observation
        idx = replbuffer.storeFrame(last_obs)
        q_input = replbuffer.encodeRecentObservation()

        #It needs to try out new action to check randomly if it can get a better reward
        if (np.random.random() < exploration.value(t)) or not model_initialized:
            action = env.action_space.sample()
        else:
            # chose action according to current Q and exploration
            action_values = session.run(q, feed_dict={obs_t_ph: [q_input]})[0]
            action = np.argmax(action_values)
        
        #move one step forward
        newState, reward, done, info = env.step(action)
        #save the transition
        replBuffer.storeEffect(idx, action, reward, done)
        lastObs = newState

        #if done, it means the end of episode
        if done:
            lastObs = env.reset()
        
        #if replayBuffer has enough samples, train the network
        if (t % learningFreq == 0 and replayBuffer.canBeSampled(batchSize)):
            obsBatch, actionBatch, rewardBatch,nextObsBatch, doneBatch = 
            replayBuffer.sample(batchSize)

            #initialize the model if it has not done yet
            if not modelInit:
                initIndependentVars(sesson, tf.global_variables(),
                                    {obs: obsBatch, nObs: nextObsBatch})
                modelInit = True
            
            feed_dict = {obs: obsBatch, act: actionBatch, rew: rewardBatch,
                         nObs: nextObsBatch,  done: doneBatch}
            session.run(dqnTrain, feed_dict=feed_dict)

            #peirodically update the targetQ
            numParamUpdates += 1
            if numParamUpdates % targetUpdateFreq == 0:
                session.run(targetQVarsToUpdate)
                numParamUpdates = 0 #reset
    









