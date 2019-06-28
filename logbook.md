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

**25 June 2019**  
I read an interresting paper introducint the term Professor Forcing https://arxiv.org/pdf/1610.09038.pdf. It uses a GAN-discriminator to force the hidden-states from teacher-forcing and generating a sequence of the same length to be close to each other. I am interrested in trying to replicate this paper because it uses two networks with shared weights.  In some cases the improvement was only marginal but it also had a regularizing effect on the network.  
I have to read more about how to implement validation loss in RNN and use it to see if the network is overfitting. At the moment, I am struggling to make the network output something sensical, I am not worried about overfitting my network atm.  
Other things I want to do is make the RNN I use for Trump tweets deeper and see if there is any improvements.  

But first I want to implement mini-batch-training for Vanilla RNN to speed up training.  
Considering to use this approach as guideline. https://gist.github.com/williamFalcon/f27c7b90e34b4ba88ced042d9ef33edd  
The problem is that for each sentence, we want to start with the same hidden-state. This also means if the length of different sentences/tweets are different we have to cut them off and padd them to all have the same length as the longest sentence left in our mini-batch.  

**26 June 2019**  
Note to self, when using PyTorch the batch-size is always the first dimension in any Tensor going into the network, even if batchsize is 1.  Also remember that the hidden layer does not have the same size as the input layer which was the size of the encoder.  
Also, loss-functions that takes class number instead of onehotencoding vector has size 1, not 2 – even if you train mini-batches.  
Implemented batch-training for my RNN-network, works with batch-size 1 to N.  Training with mini-batches gives a faster convergence (expected) and a smoother curve.  

Next step is to force the hidden state to be the zero-state whenever encountering a end-of-tweet character. I also introduced a beginning-of-tweet character to match the hidden state to start the sentence.  

I am going away from global optimizer and parameter variables when training. If I want to run multiple experiments I need to use different optimizers and my guess is that optimizers can only be bound to one set of parameters. I don't know what happens with the object if I try to over-write it and it will be hard to know which optimizer I am currently using linked to my current network. Hopefully this will make it easier to handle.  

**27 June 2019**  

I redid the notes on Attention with help from the 2015 paper – "Effective Approaches to Attention-based Neural Machine Translation". It's quite interresting how incomplete and wrong my first notes are. Still each itteration is a vital step towards the correct end product. Still I don't understand why the Transformer is possible to train with using parallellisation. Backpropagating through the attention means backpropagating through a RNN, you cannot parallelize that, it's a bottleneck by definition.

**28 June 2019**  

I added a function today that changes the hidden-layer in any batch during training to the zero-hidden-state (inital). It didn't give very good results. When I think about it in more detail there are multiple reasons why this is a bad idea.  
(i) changing the created hidden-state to the zero-state sends the wrong information to the model during backpropagation. If I rewrite the loss-function to be expecting the zero-state as output after the end of character-input it would make sense. And then the backpropagation would take taht into consideration.  
e.g if the y_target is '\*' we want to add that the loss should be both the difference between y and y_target + the difference between the output-state and the zero-state. I don't know how to make a conditional loss-function in pytorch but It's an interresting challenge.  
(ii) the goal of creating a RNN that outputs Trump tweets is to create one tweet, not multiple. We don't need to teach the model to end the tweet – at least not when playing around with the vanilla RNN.  

I also added a function to calculate the validation loss, not using batches yet. Validation-loss will be calculated from an asked for number of tweets limited by an asked for number of character each. Different from training the network which is done with tweets of longer lengths.  

What I should do is to change batch-training such that the zero-vector is invoked at the beginning of a batch itteration more times than one (which at the moment is only done one time when training the very first batch)  

! Almost forgot to mention: In pytorch we can change the value of variables without breaking the link to the networks-graph by invokeing variable.data = new_value. If we break the link by changing a value, we will break the backproagation calculating the gradient.   

**29 June 2019** 
