'''
This script is made based on the Keras tutorial https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

User can run this code by calling
python lstm_enc_decoder.py dataset.txt [--load_model LOAD_MODEL]
                                       [--train_model TRAIN_MODEL]
                                       [--batch_size BATCH_SIZE]
                                       [--epochs EPOCHS]
                                       [--latent_dim LATENT_DIM]
                                       [--max_n_samples MAX_N_SAMPLES]
                                       [--validation_split VALIDATION_SPLIT]
                                       [--dropout DROPOUT]

'''

from __future__ import print_function

import os
import sys
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
    batch_size = args.batch_size        # Batch size for training.
    epochs = args.epochs                # Number of epochs to train for.
    latent_dim = args.latent_dim        # Latent dimensionality of the encoding space.
    num_samples = args.n_samples    # Number of samples to train on.
    data_path = args.data_path          # Path to the data txt file on disk.
    validation_split = args.validation_split  # Validation_split ratio
    load_model = args.load_model        # Path to the model file saved.     
    train_model = args.train_model      # Path to the model file saved.
    dropout = args.dropout              # Dropout rate
    save_graph_path = os.path.splitext(str(train_model))[0] + '.png'
    
    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        
    actual_num_samples = min(num_samples, len(lines) - 1)
    
    for line in lines[:actual_num_samples]:
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

        # Run training
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', s2s_accuracy])

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
    val_end = min(val_start + 20, actual_num_samples)
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
    
    parser.add_argument('data_path', type=str, help='input data path')
    parser.add_argument('--load_model', type=str, help='path of model saved')
    parser.add_argument('--train_model', type=str, help='path of model to be saved')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--latent_dim', type=int, default=256, help='latent_dim inside lstm')
    parser.add_argument('--n_samples', type=int, default=10000, help='first n number of samples(lines) to be inputed for training')
    parser.add_argument('--validation_split', type=float, default=0.1, help='portion reserved for validation (last part of input data)')
    parser.add_argument('--dropout', type=float, default=0, help='droupout rate')
    return parser.parse_args(argv)



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))