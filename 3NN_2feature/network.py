from __future__ import print_function

import tensorflow as tf
import sys
import shutil
import os
import random
import numpy as np
from collections import deque

sys.path.append("game/")
import wrapped_flappy_bird as game



GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
output_directory = 'TensorBoard_logs'
TRAINRANGE = 3000000
LEARNING_RATE= 1e-4

#create network
def dqn():

    # set placeholders
    with tf.name_scope('input'):
        s = tf.placeholder("float", [None, 8],name='input')
        a = tf.placeholder("float", [None, ACTIONS],name='q')
        y = tf.placeholder("float", [None],name='y')    

    # layer 1
    with tf.name_scope('Layer_1'):
        W_fc1 = tf.Variable(tf.truncated_normal([8,128], stddev=0.1)) 
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[128])) 
        h_fc1 = tf.nn.relu(tf.matmul(s, W_fc1) + b_fc1)
    
    # layer 2
    with tf.name_scope('Layer_2'):
        W_fc2 = tf.Variable(tf.truncated_normal([128,512], stddev=0.1)) 
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[512])) 
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    
    with tf.name_scope('Layer_3'):
        W_fc3 = tf.Variable(tf.truncated_normal([512,128], stddev=0.1)) 
        b_fc3 = tf.Variable(tf.constant(0.1, shape=[128])) 
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)


    # Readout layer
    with tf.name_scope('Readout_Layer'):
        W_fc4 = tf.Variable(tf.truncated_normal([128,ACTIONS], stddev=0.1))
        b_fc4 = tf.Variable(tf.constant(0.1, shape=[ACTIONS]))

    # CNN output
    with tf.name_scope('Final_matmul'):
        readout = tf.matmul(h_fc3, W_fc4) + b_fc4
   
    # Cross entropy functions
    with tf.name_scope('Loss'):
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)  
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(tf.square(y - readout_action))
    tf.summary.scalar('Loss', cross_entropy)

    # Optimiser
    with tf.name_scope('Adam_optimizer'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    
    # Merge all the summaries and write them out to "TensorBoard_logs"
    merged = tf.summary.merge_all()

    print('Network has successfully built.')

    return s,a,y,readout,train_step,merged



# Train
def train(s,a,y,readout,train_step,merged,sess):
    
    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")


    if not os.path.exists(output_directory):
        print('\nOutput directory does not exist - creating...')
        os.makedirs(output_directory)
        print('Output directory created.')
    else:
        print('\nOutput directory already exists - overwriting...')
        shutil.rmtree(output_directory, ignore_errors=True)
        os.makedirs(output_directory)
        print('Output directory overwitten.')

    train_writer = tf.summary.FileWriter(output_directory, sess.graph)

    # open up a game state to communicate with emulator 
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    feature,r_0,terminal,RESULT = game_state.frame_step(do_nothing)
    s_t = list(feature.values())
    s_t = s_t + s_t + s_t + s_t


    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    Final = 0


    for t in range(TRAINRANGE):
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]},session=sess)[0]

        a_t = np.zeros([ACTIONS])

        action_index = 0

        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
               print("----------Random Action----------")
               action_index = random.randrange(ACTIONS)
               a_t[random.randrange(ACTIONS)] = 1
            else:
               action_index = np.argmax(readout_t)
               a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

    
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        feature_1,r_t,terminal,RESULT = game_state.frame_step(a_t)
        s_t1 = list(feature_1.values())
        s_t1 = s_t1 + s_t[0:6] 

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        
        if len(D) > REPLAY_MEMORY:
           D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch,},session=sess)
            
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

        # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch,
                }
             )
            
            # TensorBoard output evry 10 steps
            if t % 100 == 0 :
                summary, = sess.run([merged],feed_dict = {y : y_batch,a : a_batch,s : s_j_batch } )
                train_writer.add_summary(summary, t)
            
        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

       
        if Final <= RESULT:
            Final = RESULT

        # printing
        print("TIMESTEP", t, "STATE", state, \
                "EPSILON", epsilon, "ACTION", action_index, "REWARD", r_t, \
                "Q_MAX %e" % np.max(readout_t),'Q_MIN %e' % np.min(readout_t),"Result",str(Final))

    return print("The Result is",str(Final),"!")


def playGame():
    sess = tf.InteractiveSession()
    s,a,y,readout,train_step,merged = dqn()
    train(s,a,y,readout,train_step,merged,sess)

def main():
    playGame()

if __name__ == "__main__":
    main()