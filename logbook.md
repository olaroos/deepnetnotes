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

Listened to lesson 8 of fastai. Good stuff, I think I didn't know some basics as well as I should.  
I am interrested in trying to build a custom loss function that calculates loss depending on both the output and the hidden state.  

A[...,1] – "..." – symbolizes all the ranks of the matrix we haven't choosen in the same way as not selecting the last ranks of a matrix leaves them out.  
Squeeze     a tensor removes all the dimensions that has a length of 1.  
UnSqueezing a tensor adds a dimension with a length of 1.  
Flatten     a tensor to reshape it to have exactly one dimension by concatenating its rows.  

Use package torchviz make_dot() function to create an image of Pytorchs execution graph. 

**30 June 2019**  

I started writing DataSet and DataLoader functions for my RNN experiment. I introduce new terms: parentbatch is a batch containing a number of subbatches. Each parentbatch starts training with the zero-vector hidden state. Each parentbatch has a number of subbatches equal to the longest string in the parentbatch divided by the sequence-length rounded upwards. I wrote a function to create parent-batches which padded the strings with a selectable char-token such that all the subbatches are of equal length.  

Tomorrow I plan to implement training of padded batches.  

I might have to nest the DataLoader for the subbatches inside a DataLoader for the ParentBatches. more to come... 

**1 July 2019**  

I wrote a DataLoader for the parentBatches, an Itterator for the subBatches and a TwitterDataSet creator. I think I have to add more __functions__ to the subBatches itterator. I choose to make it an itterator instead of a generator because itterating over an itterator does not throw StopIteration error.  
https://www.freecodecamp.org/news/how-and-why-you-should-use-python-generators-f6fb56650888/  
https://hackaday.com/2018/09/19/learn-to-loop-the-python-way-iterators-and-generators-explained/  

Honestly, a DataLoader at this time is not necessary because the tweets are not that big and they are already loaded to the memory. In the future I guess I would want the DataLoader to subsequentually read e.g pickled data that has already been prepared from the disk.  

Preparing parentbatches of four years of Trump tweets takes almost 30 seconds, I might have refactor that function in the future.  

I also read the article about padding input for LSTMs. One of the differences doing that for LSTM compared to vanilla-RNN seem
to be that a forward-pass through a LSTM only required one forward-pass. Not a number of forwardpasses equal to the longest input-string in my RNN.  

https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e  

Why can't we just stop feeding the network for the strings that do not have any characters left? Why do we need to pad it? Good question. Don't really get why, we should be able to just feed the RNN a smaller batch. In the case of the LSTM, this might not be possible because of the way it is written in pytorch. The way the LSTM seems to be written we get as output a tensor which contains all the predictions for all the input characters. That's why we need to discard the ones that should not have been activated. 

THERE IS SO MUCH GOING ON IN this tutorial, I need some more time to figure this out. I keep my thought written down here anyway:  

"I thought about how the loss-function works and how I can adjust my functions to take padded-input and target-data. This is my guess on how it works:  
The execution-graph will save the values in the forward-pass and because we are summing up the activations and the loss and averaging them for the gradient we cannot just break the chain of the forwardpass. There are two things that need to happen if we want to train padded strings for a batch update:  
(i) The activations from the input corresponding to the padded characters should not be taken into account when calculating the gradient. How do we accomplish this? The gradient will be calculated differently depending on how the RNN is built.  
We don't want the change in the weights or the bias here to be summed up, and we want to do this by only changing the inputu x for those characters that are missing. I don't know if that is possible without changing the RNN-class function.  
proposals: put the weights.data and the biases.data values temporarily to 0?  
(ii) The calculated loss for the padded target characters should not be taken into account. This should also be set to 0 before summing up the loss.  

So the padding is done such that the functions torch.nn.utils.rnn.pack_padded_sequence and  torch.nn.utils.rnn.pad_packed_sequence can be used before and after the data is put into the LSTM unit. They are re-padded after passing through the LSTM layer. For my RNN, I should be able to do this much easier than the LSTM. As long as the shorter strings that I feed to the network do actually calculate the gradient even though I prepare to cut them out of the forwardpass when they end. I could use the printing of the execution graph to examin if that is the case" 

For some reason I thought that the loss-function had some memory or hidden function that calculated the gradient. This is not true, the gradient is only dependent on the variables connected to the execution-graph. As long as the input to the loss-functions are variables their calculated values will be connected to the execution-graph.  

Writing a custom loss-function requires one forward and one backward-function.  

**2 July 2019**  

I did an experiment where I cut down the input from 2 to 1 in the middle of training my RNN-network and created a picture of the graph. I did the same training without cutting down the input-shape and compared the graphs. Here I learned that the execution-graph remembers the slicing, there should not be any problem changing the size of input mid-training.  

I also examined the execution-graph of a summation of the loss from multiple passes in the RNN. Here it seems like no information is lost, the execution-graph remembers and will calculate the gradient with respect to each forward-pass's loss.  

link to my experiment -> https://github.com/olaroos/RNNexp/blob/master/exgraphexp.ipynb

So, now I understand how the padding in the LSTM example works. The input is cut mid-training. The only reason it is repadded again after the LSTM-pass is that the loss-function can't take input with shape that represents sequences of different lengths. Also it was noted in the comments of that article that the loss-function used has a ignore-index flag which makes the third step much less complicated.  

Also I learned that we can assign a hook to variables that requires gradient to get their gradient with respect to the loss. variable.register_hook(lambda grad: print(grad))  

