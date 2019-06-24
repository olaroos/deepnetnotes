**23 June 2019**  
I copied a tutorial on vanilla RNN. The problem I spent a long time trying to solve was to backpropagate from all the losses I create when I feed my network a number of single encoded letters. In the first tutorial, only the last loss was used for backpropagation, then I found a second tutorial in which the loss was summed together and then that element was used for backpropagation. My fear was that only the last state would have been updated if I didn't do this or that only parts of the information was used to learn the sequence.  

I don't fully understand how the backprop works in PyTorch and this might be a problem that I come back to in the future when I build more complex models.  

I want to test building many-to-one vanilla RNNs and compare the results with the one-to-one RNN I build today which was a copy of the last assignment for the DNN course I did in Matlab.

Next target is to use an optimizer, now I am doing SGD.

first tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html  
second tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#Creating-the-Network  

I found a good template for future pytorch experiments. https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139

**24 June 2019**  
I added some more specifics on LSTM and GRU. Read an article explaining GRU in more detail, LSTM uses 4 gates and GRU uses 2 gates. The problem with exploding gradient seem to be solvable by clipping the gradient, the vanishing gradient problem seem to be a harder problem to solve.  
LSTM solves it by saving long-term information in a memory-cell that is updated without multiplication with the weights. This means (how I understand it) the memory-cell can carry information very far back for the weights to be updated with. It's not that the cell itself is changed, the long term information that is saved in the LSTM is still only held in the weights of the network but the information is carried backwards with the help of the memory-cell in the LSTM during training.  

