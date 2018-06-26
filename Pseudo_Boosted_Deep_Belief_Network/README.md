# Pseudo_Boosted_Deep_Belief_Network
Source code for paper "Pseudo Boosted Deep Belief Network", ICANN 2016, Authors: Tiehang Duan; Sargur N. Srihari;

The source code is written in matlab and everything is implemented from scratch. The source code is tested on Matlab 2015a.

To run the model, please:

    1) Download the MNIST dataset (four files: t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte, train-images.idx3-ubyte, train-labels.idx1-ubyte) from Yann LeCun's Website (http://yann.lecun.com/exdb/mnist/).
    
    2) Run main.m.
    
The parameters in main.m is set for fast demo purpose. To achieve results comparable to current state-of-the-art, please set time1=10, time2=5, Iter_size=60000 in main.m.

Functions for processing the data including loadMNISTImages.m, loadMNISTLabels.m and converter.m are adopted from Ruslan Salakhutdinov and Geoff Hinton's DBN package. discretesample.m is adopted from Dahua Lin's repository.

This project is completed two years ago and is not actively maintained. 
