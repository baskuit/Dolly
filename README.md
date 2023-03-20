## About

This repo is an application of a regularized Nash dynamics on pokemon showdown. The representation code is done with a gutted version of [Poke-Env](https://github.com/hsahovic/poke-env) and the training loop is based on [Monobeast](https://github.com/facebookresearch/torchbeast).

This project is no longer being developed or tested. The time-scale necessary to test the efficacy of the neural network design or hyperparameters is simply too large. I will return to this project after my compute budget increases and when faster simulators become available.

## NN Architecture

The design of the neural network is based on Transformer Encoders. Just as convolutional nets are equivariant to the natural symmetries of image data (translation), 'BasketNet' is equivariant w.r.t. to permutation of bench pokemon. Again, I was not able to adequately test this design, but I have confidence that NLP inspired architectures are much more suited to this task than say, a fully connected network.
