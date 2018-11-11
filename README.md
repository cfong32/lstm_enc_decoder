# lstm_enc_decoder

Here a sequence-to-sequence model of a simple LSTM encoder-decoder architecture is implemented.
The code is modified from the Keras tutorial https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

User can run this code by calling
python lstm_enc_decoder.py dataset.txt [--load_model LOAD_MODEL]
                                       [--train_model TRAIN_MODEL]
                                       [--batch_size BATCH_SIZE]
                                       [--epochs EPOCHS]
                                       [--latent_dim LATENT_DIM]
                                       [--max_n_samples MAX_N_SAMPLES]
                                       [--validation_split VALIDATION_SPLIT]
                                       [--dropout DROPOUT]

