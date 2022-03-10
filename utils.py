import tensorflow as tf
import logging
import numpy as np


def get_chunks(l, batch_size):
    return [l[offs:offs + batch_size] for offs in range(0, len(l), batch_size)]


def tf_argchoice_element(sequence, element):
    random_uniform = tf.random.uniform(shape=tf.shape(sequence))
    y = tf.ones(shape=tf.shape(sequence)) * -1
    random_uniform = tf.where(tf.equal(sequence, element), x=random_uniform, y=y)
    idx = tf.argmax(random_uniform, axis=-1, output_type=tf.int32)
    return idx
def tf_argchoice_element2(sequence, valid_words, element, element2):
    random_uniform = tf.random.uniform(shape=tf.shape(sequence))
    y = tf.ones(shape=tf.shape(sequence)) * -2
    random_uniform = tf.where(tf.equal(sequence, element), x=random_uniform, y=y)
    y = tf.ones(shape=tf.shape(sequence)) * -1
    random_unifrom2 = tf.where(tf.equal(valid_words, element2), x =random_uniform, y = y  )
    y = tf.ones(shape=tf.shape(sequence)) * -2
    random_unifrom2 = tf.where(tf.equal(sequence, element), x=random_unifrom2, y=y)
    idx = tf.argmax(random_unifrom2, axis=-1, output_type=tf.int32)
    logging.info(random_unifrom2.shape)
    return idx
'''
def tf_argchoice_element2(sequence, valid_words , element):
    random_uniform = tf.random.uniform(shape=tf.shape(sequence))
    y = tf.ones(shape=tf.shape(sequence)) * -1
    random_uniform = tf.where(tf.equal(sequence, element) , x=random_uniform, y=y)
    random_uniform = tf.where(tf.equal(valid_words, True ), x = random_uniform, y = random_uniform )
    idx = tf.argmax(random_uniform, axis=-1, output_type=tf.int32)
    logging.info("got the insert idx")
    return idx
'''


def get_embeddings(s2v_file):
    embeddings = np.load(s2v_file)
    logging.info('loaded {} embeddings with dimension {} from npy file'.format(embeddings.shape[0],
                                                                               embeddings.shape[1]))
    return embeddings
