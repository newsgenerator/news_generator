#coding=utf-8
import tensorflow as tf
import Discriminator
import Gennerator
from process_data import input_data
import sys, os
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
import tflib as lib
import numpy as np
import time
import tflib.ops.linear
import tflib.ops.conv1d

BATCH_SIZE = 64
SEQ_LEN = 2
SCORE_DIM = 20  #generator score dimention
DIM = 128
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
ITERS = 2000
CRITIC_ITERS = 10

data = input_data()
charmap = data._charmap

def train():
    score_for_gen = tf.placeholder(tf.float32, shape=[None, SCORE_DIM])
    score_for_disc = tf.placeholder(tf.float32, shape=[None, 2, len(charmap)])
    real_inputs_discrete = tf.placeholder(tf.int32, shape=[None, SEQ_LEN])
    real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
    fake_inputs = Gennerator.Gennerator(BATCH_SIZE, score_for_gen)
    fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)
    
    sample_size = tf.placeholder(tf.int32, shape=[])
    sample_inputs = Gennerator.Gennerator(sample_size, score_for_gen)
    
    
    
    disc_real = Discriminator.Discriminator(real_inputs, score_for_disc) 
    disc_fake = Discriminator.Discriminator(fake_inputs, score_for_disc)
    
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)
    
    
    # WGAN lipschitz-penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha*differences)
    gradients = tf.gradients(Discriminator.Discriminator(interpolates, score_for_disc), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty
    
    #add summary info
    tf.summary.scalar('Gennerator Cost',gen_cost)
    tf.summary.scalar('Discriminator Cost', disc_cost)
    
    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')
    
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

    #summary initializer
    merged = tf.summary.merge_all()
    with tf.Session() as session:

        summary_writer = tf.summary.FileWriter('logs/',session.graph)
        session.run(tf.global_variables_initializer())
        sc_gen = None 
        sc_disc = None
        _data = None
        
        def generate_samples():
            sc_gen, sc_disc, _data = data.get_next_batch(len(data._score_test), 0)
            samples = session.run(sample_inputs, feed_dict={sample_size: len(sc_gen), score_for_gen:sc_gen, score_for_disc:sc_disc})
            samples = np.argmax(samples, axis=2)
            decoded_samples = []
            for i in range(len(samples)):
                decoded_samples.append((data._inv_charmap[samples[i][0]], data._inv_charmap[_data[i][0]]))
            return decoded_samples

        batch_index = 0
        for iteration in range(ITERS):

            # Train generator
            if iteration > 0:
                _ = session.run(gen_train_op, feed_dict={score_for_gen:sc_gen, score_for_disc:sc_disc})


             # Train critic
            for i in range(CRITIC_ITERS):
                if batch_index * BATCH_SIZE > len(data._score):
                    batch_index = 0
                sc_gen, sc_disc, _data= data.get_next_batch(BATCH_SIZE, batch_index)

                batch_index += 1
                summary,_disc_cost, _gen_cost, _gradient_penalty,  _ = session.run(
                    [merged,disc_cost, gen_cost, gradient_penalty, disc_train_op],
                    feed_dict={real_inputs_discrete:_data, score_for_gen:sc_gen, score_for_disc: sc_disc}
                )
                summary_writer.add_summary(summary)

            #lib.plot.plot('time', time.time() - start_time)
            #lib.plot.plot('train disc cost', _disc_cost)

            #if iteration % 100 == 99:
                #lib.plot.flush()

            #lib.plot.tick()
            if iteration % 10 == 0:
                print(iteration, _disc_cost, _gen_cost, _gradient_penalty)

            if iteration % 100 == 0:
                sample = generate_samples()
                for i in range(len(sample)):
                    print(data._score_test[i], sample[i])

            if iteration % 500 == 0:
                saver = tf.train.Saver()
                saver.save(session, 'models/trained_model_',global_step=iteration)
    summary_writer.close()
            
    
    sess=tf.Session()   
    saver = tf.train.import_meta_graph('trained_model_-1500.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    
    #test one 
    test_1_score = [(120, 60)]
    #whatever
    how_win_1 = ['大胜']
    one_dict = {}
    for i in range(100):
        sc_gen, sc_disc, _data = data.get_next_batch(len(test_1_score), 0)
        sample = sess.run(sample_inputs, feed_dict={sample_size: len(sc_gen), score_for_gen:sc_gen, score_for_disc:sc_disc})
        sample2 = np.argmax(sample, axis=2)
        hw = data._inv_charmap[sample2[0][0]]
        one_dict[hw] = one_dict[hw] + 1 if hw in one_dict else 1
    """
    print(sc_disc[0][0][0], sc_disc[0][1][0])
    print(sample[0][0])
    print(sample2[0][0])
    print(inv_charmap[sample2[0][0]])
    inv_charmap
    """
    one_dict
    
    
    #test one 
    test_1_score = [(120, 118)]
    #whatever
    how_win_1 = ['险胜']
    one_dict = {}
    for i in range(100):
        sc_gen, sc_disc, _data = data.get_next_batch(len(test_1_score), 0)
        sample = sess.run(sample_inputs, feed_dict={sample_size: len(sc_gen), score_for_gen:sc_gen, score_for_disc:sc_disc})
        sample2 = np.argmax(sample, axis=2)
        hw = data._inv_charmap[sample2[0][0]]
        one_dict[hw] = one_dict[hw] + 1 if hw in one_dict else 1
    """
    print(sc_disc[0][0][0], sc_disc[0][1][0])
    print(sample[0][0])
    print(sample2[0][0])
    print(inv_charmap[sample2[0][0]])
    inv_charmap
    """
    one_dict
    
train()