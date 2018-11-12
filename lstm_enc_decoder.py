'''
Here is implemented a simple (single-layer) LSTM encoder-decoder sequence-to-sequence model (LSTM_ED_S2S).
It aims to provide a simple command line environment for user to test LSTM_ED_S2S model on their dataset more easily.
The code is modified upon the Keras tutorial https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

User can run this code by calling
python lstm_enc_decoder.py INPUT_DATA.txt [--load_model PATH_TO_LOAD]
                                          [--train_model PATH_TO_SAVE]
                                          [--learning_curve PATH_TO_SAVE(default: 'learning_curve.png')]
                                          [--batch_size INTEGER(default: 64)]
                                          [--epochs INTEGER(default: 100)]
                                          [--latent_dim INTEGER(default: 256)]
                                          [--n_samples INTEGER(default: 1000)]
                                          [--test_split_ratio FLOAT(default: 0.1)]
                                          [--dropout_rate FLOAT(default: 0)]
'''

from __future__ import print_function

import os
import sys
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from visual_callbacks import AccLossPlotter

def s2s_accuracy(y_true, y_pred):
    print (y_true.shape)
    print (y_pred.shape)
    print (y_true)
    print (y_pred)
    
    return 'a'
    return K.zeros(y_pred)

def main(args):
    # Reading input arguments
    batch_size = args.batch_size  # Batch size for training.
    epochs = args.epochs          # Number of epochs to train for.
    latent_dim = args.latent_dim  # Latent dimensionality of the encoding space.
    num_samples = args.n_samples  # Number of samples to train and test on.

    data_path = args.input_data           # Path to the data txt file on disk.
    load_model = args.load_model          # Path to the model file saved.     
    train_model = args.train_model        # Path to the model file saved.
    save_graph_path = args.learning_curve  # Path to save the graph
    if save_graph_path[-4:] != '.png': save_graph_path += '.png'

    validation_split = args.test_split_ratio  # Validation_split ratio
    dropout = args.dropout_rate               # Dropout rate

    num_test_output = args.n_test_output      # Number of test outputs


    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        
    actual_num_samples = min(num_samples, len(lines) - 1)
    shuffled_lines = lines[:actual_num_samples].copy()
    random.shuffle(shuffled_lines)

    for line in shuffled_lines:
        input_text, target_text = line.split('\t')
        
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True, dropout=dropout)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    
    # Load trained model weights
    if load_model:
        model.load_weights(load_model)
    
    # Train, then save model
    history = None
    if train_model:
        
        # To visualize learning curve
        plotter = AccLossPlotter(graphs=['loss', 'acc'], save_graph=save_graph_path)

        # Define custom accuracy metric
        # It simply not to compare the character after '\n' (suppose to be the last character)
        def acc(y_true, y_pred):
            y_true_copy = y_true[:,:-1,:]
            y_pred_copy = y_pred[:,:-1,:]
            return K.cast(K.equal(K.argmax(y_true_copy, axis=-1), K.argmax(y_pred_copy, axis=-1)), K.floatx())

        # Run training
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=[acc])

        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=validation_split,
                            callbacks=[plotter])
        # Save model
        model.save(train_model)
    
        
    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
               len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence


    val_start = int(actual_num_samples*(1-validation_split))
    val_end = min(val_start + num_test_output, actual_num_samples)
    for seq_index in range(val_start, val_end):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input:', input_texts[seq_index])
        print('True result:', target_texts[seq_index][1:-1])
        print('Predicted result:', decoded_sentence)
    
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_data', type=str, help='path of the input data, normally a txt file')

    parser.add_argument('--load_model', type=str, help='path of the saved .h5 model to load')
    parser.add_argument('--train_model', type=str, help='path of the .h5 model to save to')
    parser.add_argument('--learning_curve', type=str, default='learning_curve.png', help='path to save the learning curve plot')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension inside the LSTM')
    parser.add_argument('--n_samples', type=int, default=1000, help='first n number of samples(lines) from the input data to be loaded for training and testing')

    parser.add_argument('--test_split_ratio', type=float, default=0.1, help='portion of the dataset reserved for testing')
    parser.add_argument('--dropout_rate', type=float, default=0, help='droupout rate')

    parser.add_argument('--n_test_output', type=int, default=20, help='number of prediction results (from the test set) to be printed out')
    return parser.parse_args(argv)



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))