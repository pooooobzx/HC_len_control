import collections
from itertools import combinations

import numpy as np
import logging
import tensorflow as tf
from utils import tf_argchoice_element, tf_argchoice_element2


class State(
        collections.namedtuple("State", ("state", "internal_state"))):
    pass


def get_extractive_next_state_func(batch_size, sentence_length, summary_length, sentence,char_length, char_size_sen,char_size_sen_batch,helper):

    def get_next_state(state):
        boolean_map = state.internal_state
        remove_idx = tf_argchoice_element(boolean_map, element=tf.constant(True, dtype=tf.bool))
        char_size_sen_batch = tf.broadcast_to(char_size_sen, shape=[batch_size, sentence_length])
        

        remove_mask = tf.cast(tf.one_hot(remove_idx, sentence_length), dtype=tf.bool)
        next_boolean_map = tf.math.logical_xor(boolean_map, remove_mask)


        
        d = next_boolean_map
        y = tf.zeros(shape=tf.shape(next_boolean_map), dtype = tf.int32)
        logging.info(char_size_sen_batch.shape)
        logging.info(y.shape)
        g = tf.where(tf.equal(d, True), char_size_sen_batch, y)
        
        selected_len = tf.math.reduce_sum(g, 1, keepdims=True)
        durable = tf.subtract(char_length, selected_len)
        

        durable = tf.broadcast_to(durable, shape=[batch_size, sentence_length])
        #durable = tf.tile(durable, (1,))
        falses = tf.zeros(shape=tf.shape(char_size_sen_batch), dtype = tf.bool)
        trues = tf.ones(shape=tf.shape(char_size_sen_batch), dtype=tf.bool)
 
        #durable= tf.repeat(durable, repeats = tf.shape(char_size_sen), axis = 1 )
        boolean_map2 = tf.where(tf.math.less_equal( char_size_sen_batch, durable), trues, falses )
        insert_idx = tf_argchoice_element2(next_boolean_map, boolean_map2, element=tf.constant(False, dtype=tf.bool), element2=tf.constant(True, dtype=tf.bool))
        insert_mask = tf.cast(tf.one_hot(insert_idx, sentence_length), dtype=tf.bool)
        
        next_boolean_map2 = tf.math.logical_or(next_boolean_map, insert_mask)

        sequence = tf.broadcast_to(sentence, shape=[batch_size, sentence_length])
        flat_output = tf.boolean_mask(sequence, next_boolean_map2)
        next_state = tf.reshape(flat_output, shape=(batch_size, summary_length))

        next_state = State(state=next_state, internal_state=next_boolean_map2)

        return next_state

    return get_next_state


def get_extractive_initial_states(num_restarts, batch_size, x_sentence, summary_length,sentence, char_length, config, exhaustive=False):
    summary_length = int(summary_length)
    char_length=int(char_length)
    sentence_length = x_sentence.size
    sentence_char_len = sum(map(lambda x: len(x), sentence)) + len(sentence) - 1
    def yield_batch(boolean_maps, states):
        initial_state = State(state=np.asarray(states), internal_state=np.asarray(boolean_maps))
        return initial_state

    boolean_maps = list()
    states = list()

    if sentence_length <= summary_length:
        states = [x_sentence]
        boolean_maps = [[True for _ in range(sentence_length)]]
    elif exhaustive:
        skipgrams = set(combinations(x_sentence, int(summary_length)))
        for skipgram in skipgrams:
            boolean_map = np.zeros(shape=sentence_length, dtype=np.bool)
            boolean_maps.append(boolean_map)
            states.append(skipgram)

            if len(states) == batch_size:
                yield yield_batch(boolean_maps, states)
                boolean_maps = list()
                states = list()
    else:
        mylen = np.vectorize(len)
        repeat  = 0 
        potential = [summary_length -1 , summary_length, summary_length+1]
        for _ in range(num_restarts):
            
            summary_length=potential[repeat % 3 ]

            config["summary_length"] = summary_length
            boolean_map = np.zeros(shape=sentence_length, dtype=np.bool)
            idx_positive = np.random.choice(range(sentence_length), size=summary_length, replace=False)
            boolean_map[idx_positive] = True
            sentence_array = np.asarray(sentence)
            selected = sentence_array[np.where(boolean_map)]
            selected_char_len = int(np.sum(mylen(selected))) + len(selected) -1
            i = 0
            while selected_char_len > char_length : #give a time limit
                logging.info("looking for proper length {}".format(summary_length))
                i += 1 
                if i == 101:
                    logging.info("settle for a non-match len")
                    break
                boolean_map = np.zeros(shape=sentence_length, dtype=np.bool)



            
                #idx_positive = np.random.choice(list(set(range(sentence_length)) - set(np.array(np.where(boolean_map)).tolist()[0])), size=1, replace=False)
                idx_positive = np.random.choice(range(sentence_length) , size=summary_length, replace=False)
                boolean_map[idx_positive] = True
                selected = sentence_array[np.where(boolean_map)]
                selected_char_len = int(np.sum(mylen(selected))) + len(selected) -1
            logging.info("selected char len is {}".format(selected_char_len))
            boolean_maps.append(boolean_map)

            state = x_sentence[np.where(boolean_map)]
            states.append(state)

            if len(states) == batch_size:
                yield yield_batch(boolean_maps, states)
                repeat += 1 
                boolean_maps = list()
                states = list()
    if len(states) > 0:
        yield yield_batch(boolean_maps, states)
