**23 June 2019**  
I copied a tutorial on vanilla RNN. The problem I spent a long time trying to solve was to backpropagate from all the losses I create when I feed my network a number of single encoded letters. In the first tutorial, only the last loss was used for backpropagation, then I found a second tutorial in which the loss was summed together and then that element was used for backpropagation. My fear was that only the last state would have been updated if I didn't do this or that only parts of the information was used to learn the sequence.  

I don't fully understand how the backprop works in PyTorch and this might be a problem that I come back to in the future when I build more complex models.  

I want to test building many-to-one vanilla RNNs and compare the results with the one-to-one RNN I build today which was a copy of the last assignment for the DNN course I did in Matlab.

Next target is to use an optimizer, now I am doing SGD.

first tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html  
second tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#Creating-the-Network  

**24 June 2019**  
