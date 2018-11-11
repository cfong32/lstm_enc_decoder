# lstm_enc_decoder
Here is implemented a simple (single-layer) LSTM encoder-decoder sequence-to-sequence model (LSTM_ED_S2S).
It aims to provide a simple command line environment for user to test LSTM_ED_S2S model on their dataset more easily.
The code is modified upon the Keras tutorial https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

### Input data:
- Simple text file.
- Every line contains an input-target pair of strings seperated by a tab (\t) character.
- Please see dataset/eng_fra.txt and dataset/abc_abc.txt as examples.

### Outputs:
- It saves the trained model to a h5 file, specified in the *--train_model* argument.
- It plots and updates the learning curve and accuracy during training and afterwards saves it to a png file, specified in the *--learning_curve* argument.
- After training, from the training set as well as the test(validation) set, it prints some sampled the 1) Input, 2) Training target and Predicted results.

### How to run:
User can run this code by calling
```
python lstm_enc_decoder.py INPUT_DATA.txt [--load_model PATH_TO_LOAD]
                                          [--train_model PATH_TO_SAVE]
                                          [--learning_curve PATH_TO_SAVE(default: 'learning_curve.png')]
                                          [--batch_size INTEGER(default: 64)]
                                          [--epochs INTEGER(default: 100)]
                                          [--latent_dim INTEGER(default: 256)]
                                          [--n_samples INTEGER(default: 1000)]
                                          [--train_test_split_ratio FLOAT(default: 0.1)]
                                          [--dropout_rate FLOAT(default: 0)]
```
