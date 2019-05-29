from __future__ import print_function

import tensorflow as tf
import cv2
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
OBSERVE = 64. # timesteps to observe before training
EXPLORE = 100. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 64 # number of previous transitions to remember
BATCH = 16 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE= 1e-6
output_directory = 'TensorBoard_logs'
TRAINRANGE = 200


#create network
def dqn():

    # set placeholders
    with tf.name_scope('input'):
        s = tf.placeholder("float", [None, 80, 80, 4],name='State')
        a = tf.placeholder("float", [None, ACTIONS],name='q')
        y = tf.placeholder("float", [None],name='y')

    # set dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)

    # vistualize input_image
    with tf.name_scope('input_image'):
        tf.summary.image('input', s, 10)

    # First conv+pool layer
    with tf.name_scope('conv1'):
        with tf.name_scope('weights'):
            W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.1))
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(W_conv1)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv1 - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(W_conv1))
                tf.summary.scalar('min', tf.reduce_min(W_conv1))
                tf.summary.histogram('histogram', W_conv1)

        with tf.name_scope('biases'):
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(b_conv1)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv1 - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(b_conv1))
                tf.summary.scalar('min', tf.reduce_min(b_conv1))
                tf.summary.histogram('histogram', b_conv1)

        with tf.name_scope('Wx_plus_b'):
            preactivated1 = tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1],padding='SAME') + b_conv1
            h_conv1 = tf.nn.relu(preactivated1)
            tf.summary.histogram('pre_activations', preactivated1)
            tf.summary.histogram('activations', h_conv1)

        with tf.name_scope('max_pool'):
            h_pool1 =  tf.nn.max_pool(h_conv1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    '''
        #save output of conv layer to TensorBoard - first 32 filters
        with tf.name_scope('Image_output_conv1'):
            image = h_conv1[0:1, :, :, 0:16]
            image = tf.transpose(image, perm=[3,1,2,0])
        tf.summary.image('Image_output_conv1', image)
        #save a visual representation of weights to TensorBoard


    with tf.name_scope('Visualise_weights_conv1'):
        # We concatenate the filters into one image of row size 8 images
        W_a = W_conv1                      # i.e. [5, 5, 1, 32]
        W_b = tf.split(W_a, 32, 3)         # i.e. [32, 5, 5, 1, 1]
        rows = []
        for i in range(int(32/8)):
            x1 = i*8
            x2 = (i+1)*8
            row = tf.concat(W_b[x1:x2],0)
            rows.append(row)
        W_c = tf.concat(rows, 1)
        c_shape = W_c.get_shape().as_list()
        W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1])
        tf.summary.image("Visualize_kernels_conv1", W_d, 1024)
    '''

    # Second conv+pool layer
    with tf.name_scope('conv2'):
        with tf.name_scope('weights'):
            W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.1))
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(W_conv2)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv2 - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(W_conv2))
                tf.summary.scalar('min', tf.reduce_min(W_conv2))
                tf.summary.histogram('histogram', W_conv2)

        with tf.name_scope('biases'):
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(b_conv2)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv2 - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(b_conv2))
                tf.summary.scalar('min', tf.reduce_min(b_conv2))
                tf.summary.histogram('histogram', b_conv2)
        with tf.name_scope('Wx_plus_b'):
            preactivated2 = tf.nn.conv2d(h_pool1, W_conv2,strides=[1, 2, 2, 1],padding='SAME') + b_conv2
            h_conv2 = tf.nn.relu(preactivated2)
            tf.summary.histogram('pre_activations', preactivated2)
            tf.summary.histogram('activations', h_conv2)
        with tf.name_scope('max_pool'):
            h_pool2 =  tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    '''
        #save output of conv layer to TensorBoard - first 16 filters
        with tf.name_scope('Image_output_conv2'):
            image = h_conv2[0:1, :, :, 0:16]
            image = tf.transpose(image, perm=[3,1,2,0])
            tf.summary.image('Image_output_conv2', image)
        #save a visual representation of weights to TensorBoard


    with tf.name_scope('Visualise_weights_conv2'):
        # We concatenate the filters into one image of row size 8 images
        W_a = W_conv2
        W_b = tf.split(W_a, 64, 3)
        rows = []
        for i in range(int(64/8)):
            x1 = i*8
            x2 = (i+1)*8
            row = tf.concat(W_b[x1:x2],0)
            rows.append(row)
        W_c = tf.concat(rows, 1)
        c_shape = W_c.get_shape().as_list()
        W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1])
        tf.summary.image("Visualize_kernels_conv2", W_d, 1024)
    '''

    # Third conv+pool layer
    with tf.name_scope('conv3'):
        with tf.name_scope('weights'):
            W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(W_conv3)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv3 - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(W_conv3))
                tf.summary.scalar('min', tf.reduce_min(W_conv3))
                tf.summary.histogram('histogram', W_conv3)

        with tf.name_scope('biases'):
            b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]))
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(b_conv3)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv3 - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(b_conv3))
                tf.summary.scalar('min', tf.reduce_min(b_conv3))
                tf.summary.histogram('histogram', b_conv3)
        with tf.name_scope('Wx_plus_b'):
            preactivated3 = tf.nn.conv2d(h_pool2, W_conv3,strides=[1, 1, 1, 1], padding='SAME') + b_conv2
            h_conv3 = tf.nn.relu(preactivated3)
            tf.summary.histogram('pre_activations', preactivated3)
            tf.summary.histogram('activations', h_conv3)
        with tf.name_scope('max_pool'):
            h_pool3 =  tf.nn.max_pool(h_conv3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    '''
        # save output of conv layer to TensorBoard - first 16 filters
        with tf.name_scope('Image_output_conv3'):
            image = h_conv3[0:1, :, :, 0:16]
            image = tf.transpose(image, perm=[3,1,2,0])
            tf.summary.image('Image_output_conv3', image)

    # save a visual representation of weights to TensorBoard
    with tf.name_scope('Visualise_weights_conv3'):
        # We concatenate the filters into one image of row size 8 images
        W_a = W_conv3
        W_b = tf.split(W_a, 64, 3)
        rows = []
        for i in range(int(64/8)):
            x1 = i*8
            x2 = (i+1)*8
            row = tf.concat(W_b[x1:x2],0)
            rows.append(row)
        W_c = tf.concat(rows, 1)
        c_shape = W_c.get_shape().as_list()
        W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1])
        tf.summary.image("Visualize_kernels_conv3", W_d, 1024)
    '''

    # Fully connected layer
    with tf.name_scope('Fully_Connected'):
        W_fc1 = tf.Variable(tf.truncated_normal([256, 512], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))
        # Flatten the output of the second pool layer
        h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

        # Dropout
        #keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=1.0)


    # Readout layer
    with tf.name_scope('Readout_Layer'):
        W_fc2 = tf.Variable(tf.truncated_normal([512, ACTIONS], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[ACTIONS]))


    # CNN output
    with tf.name_scope('Final_matmul'):
        readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


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
    print('CNN successfully built.')
    return s,a,y,h_fc1,readout,train_step,merged,keep_prob


# Train
def train(s,a,y,h_fc1,readout,train_step,merged,keep_prob,sess):

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
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # start training
    epsilon = INITIAL_EPSILON
    t = 0

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
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

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
                keep_prob: 1.0 }
            )

            # TensorBoard output evry 10 steps
            if t % 10 == 0 :
                summary, = sess.run([merged],feed_dict = {y : y_batch,a : a_batch,s : s_j_batch, keep_prob: 1.0 } )
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


        # printing
        print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        a_file = open("logs_" + GAME + "/readout.txt", 'w')
        h_file = open("logs_" + GAME + "/hidden.txt", 'w')
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]},session=sess)[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)

    return print("End of training!")

def playGame():
    sess = tf.InteractiveSession()
    s,a,y,h_fc1,readout,train_step,merged,keep_prob = dqn()
    train(s,a,y,h_fc1,readout,train_step,merged,keep_prob,sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
