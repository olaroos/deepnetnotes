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

**3 July 2019**  

I rewrote the functions that create the data-batches, they produced tensors that were padded in the wrong way. Now the functions are more easy to understand. I also sort the batches of tweets the longest first. If the lengths would be randomly distributed in the batch, I could lose that information when I remove the tensors that have padded input and try to match them afterwards with the correct hidden-states after the forward-pass.  

**6 July 2019** 

Testing the make_parentbatch() function with real data and rewrote the generators for my DataLoader() function. Generators can loop through other generators, why didn't I think about that before.  

I moved the batch_strings() function inside the make_parentbatch() function because they take the same input-parameters (don't want errors because I for some reason put different parameter-values into them), they depend on eachother temporarily  and because I mixed up the batchsize and sequencelength variables inside the functions and had a hard time finding the errors.  

**7 July 2019**  

I got batch-training working finally. Protip, don't load data onto the graphics-card until you want to do training. I ran out of 11 Gb memory when I tried to preprocess 15000 trump tweets. Now I make the tensors cuda right before I do forward-pass.  

bugg_1: found out that my make_parentbatch() function sometimes creates batches that consists solely of padded elements. My guess is that this is an error in the pad() function but I didn't look into it yet.  

I started adding more layers to the RNN to see if it makes a difference. Adding an extra layer doing short training didn't improve validation accuracy but I didn't put a non-linear function between the layers yet which I believe will make a difference.

Creating DataLoaders really makes the training easier to manage. I deleted three big clumpsy functions.  

I installed the nbextensions_configurator package through conda but couldn't find any extensions to choose from in the graphical interface. I remember I added it to a previous jupyter build but this time it didn't function as before. It's not a biggie, I will probably look into it again in the future. Code folding is something I miss in jupyter and it would be nice to tweak the boring interface.  

**8 July 2019**  

Wrote a function that does a training taking in number of itterations,optimizer,model,dataloader etc because I wanted to train multiple models and compare the validation plots.  It was possible to add more layers in the RNN and I think it made a big difference. Instead of 0.2 in validation error I'm down to 0.15 with an extra linear-layer, batch-norm and a relu.  

I still need to scale the validation error to match the level of values of the training-error and plot them at the same time. I also want to change optimizer, using RMSProp atm.  

Things I should do in the near future:  
- write a dataloader for the validation set as well.  
- give the jupyter kernel more memory, I'm not sure but I guess this is the reason my notebook has frooze up several times on Rolfs computer. I deepcopied all the tweets twice and the notebook completely died.  
- write a function to easily create different architectures of RNN networks to experiment on and compare outcome.  

It might actually be easier and more memory efficient to create each batch at the moment my train functions asks for it. I could continously random select a batchsize of tweets and never have to create all batches before I start training.

**9 July 2019**  

Changed the DataLoader to not preprocess tweets and instead load each set of batches for a batch-size of tweets when needed. I also made it shuffle the tweets after each epoch. This dataloader will also never throw StopIteration. This change should solve the problem with jupyter notebook freezing.  

I also finished watching lesson 9 of fastai. Inspired by the structure they use in fastai I started changing my code because I want to implement the callback functions Jeremy uses in the future.  

**10 July 2019**  

I started doing mayor refactoring of my code, inspired by fastai's callbacks. calculating validation loss now uses the same DataLoader class as the training data. The dataloader no longer returns a bool that explains when to use the zero-hidden state. Looking at the first onehotencoded character of X in each batch will give us the same information. The begin-symbol is in the encoders second first position and hence if we find a "1" in that position we know that we should use the zero-hidden state.  

Everything is a little bit messy right now because I don't know if I want to implement the newly introduced Runner class right away or if I should implement the callbacks separately first as a middle step.  

Anyway, refactoring makes it much easier to see what is going on when training the network.  

I also spent some time installing and experimenting with _tags_ and _Ack_ (a search function like grep) for vim which I think I will definitely be using in the future. For tags to work a tags-file need to be precomputed with the command ctag(s?) in terminal. These files need to be locatable by vim, configure the vimrc-file to be able to find them.  

**11 July 2019**  

Apparently I broke the training, something is wrong when I train the network and I can no longer generate as good results as I did before. It seemed to have happened when I changed the validation-loss calculations, I will actually backtrack to a previous commit and start again. I remember what I did today and I believe I can make it better if I rewrite it from an earlier commit and verify that the training is working as I move along.  

**12 July 2019**  

So, the errors I thought were my fault was in fact encoding errors. These errors seems to happen sporadically. Sometimes when I load in the tweet-data and create my encoder/decoders the symbols is saved with the wrong encoding.  

My first guess is that this might have something to do with the extra swap I created recently, what if it was created with another encoding and when that is used the wrong encoding is saved to RAM.  

I will run some tests tomorrow (July 13).  

**13 July 2019**  

I found out the source to why my network stopped learning and it didn't have anything to do with the swap.  

The error stemmed from putting the RNN in training-mode. Before I hade been training my network in evaluation-mode. Because I implemented a batch-norm layer in my RNN it got me no results when the network was in training-mode. 

**14 July 2019**  

I decided to postpone inmplementing callbacks because I think initially it will get in the way of me experimenting with different structures of RNN, GRU and LSTM.  

I moved functions that I don't need to rewrite to separate files. I also tried using autoreload 2 but it didn't seem to work for the functions I imported.  

I also did a short experiment with extending the depth from 2 to 3 layers in the vanilla RNN and it seems to give better accuracy.  

**15 July 2019**  

The RNN I'm developing put the input and the hidden layer into two sub-networks. One for generating the new hidden layer and another to generate the next character. 

Comparing losses: vanilla-RNN with 2 and 3 (character-generating) layers the 3-layer network had 5\% less loss. Making it 4-layers improves the results even more, making it 5 layers might make a minimal difference.  

Trying to change the number of layers generating the hidden state by adding one layer also seem to make a minimal decrease of validation-loss.  

I'm planing to implement GRU as the next step. The generated sequences I get with these networks is much better than the ones I produced with Matlab code.  

My guess here is that in the matlab-assignment we used a simplification. All the errors calculated at each input step was the average of all errors. Now, my guess is that pytorch does not use the average-error even though I'm adding all the errors together from each step.  

When pytorch creates the execution-graph it remembers the amount of error each input-step generated and how much that error contributed to the total error used in the loss-function. (I should really put this to the test in some way).  

When calculating the validation-error every 100th iteration the graph is very jagged and in my experiments I have a hard time seeing if the validation-loss increases (a sign of overfitting). I tried calculating a simple-moving average and plotting it to try to notice if there was overfitting taking place but realised that I haven't trained the model long enough. I doubled the training length and it still didn't show any signs of overfitting...


**16 July 2019**  

Trained the network for almost 300.000 itterations batchsize 20 and no signs of overfitting. I will try to make the linear-layers wider – both for the hidden and the output to see if this can induce overfitting.  

I also made sure to code and understand the exponential-weighted-moving-average function with the debiasing Jeremy presents in lesson 11. The debiasing is an addon that can be used on the Adam optimizer but isn't part of the original version (as I understand it). The debiasing part is multiplication with the term (1 - beta^(i))^(-1) which will counter the inital bias produced by the momentum in the beginning of training but shrink to a value of 1 quite fast.  

I'm getting ready to implement the callbacks into the Learner() class. Initially I'm planing to use callbacks to change the learning-rate.  

**19 July 2019**  

I started implementing callbacks and I think that it's going to be easier to debugg the training in the future. I only got to get the default working first.  

I'm opposed to save the variable that tells the network to stop training in CallbackHandler, it would be better to put it in learner because I cannot access it from the individual callback objects.  

I created a debugging ipynb file and I'm using it to find a bugg where batches are filled with only padded elements which breaks my training.  

A stats struct has been added to the learner which will hold the graph data etc.  

**20 July 2019**  

Fixed the bugg that creates batches of only pad-characters. Moved some variables from CallbackHandler to Learner-classobjects. Started writing code for the GRU unit.  

Up until now I thought that I was inheriting from a RNN module, but I wasn't. The RNN-unit I was using was completely written by myself. I'm going to do the same with the GRU and the LSTM unit. Because if I know how to write them it's going to be much faster training them by implementing them in Swift in the future. And I can hopefully play around with different tweaks in the meantime even if training-time is not optimal.  

**22 July 2019**  

GRU-module is now trainable.  

**23 July 2019**  

Made functions to build a specialised schematic for changing selectable parameters during training steps. Improved the callback-functions. GRU doesn't seem to converge as fast as my vanilla-RNN does. Need to do longer trainingruns with the GRU and build a function to analyse the gradients to see if exploding gradient is a problem in my GRU or RNN networks. I should introduce gradient clipping in some way.  

**25 July 2019**  

Started writing the stacked gru module. I realize I have to rewrite the model.parameters() function. My SGRU-module is made out of a number of single GRU-modules. I have to figure out what structure the optimizer expects when feeding it the parameters of my module.   

It seems like the optimizer expects a generator, I will try to loop through all the single GRU-modules and yield the output 
from their parameters() function:  

def parameters(self):
  for stack in range(n_stacks):
    for param in iter(self.GRUs[stack].parameters()):
      yield param

**26 July 2019**  

I made the SGRU work. I also noticed an error that keeps coming back. It seems like I have to define the function that uses torch.cuda.is_available() in each module-file I want to use it in. if I make these module files inherit this function from another module-file it seems CUDA casts an error.  

When I was writing the Stacked GRU function pythorch threw an error where it said I updated a variable used for calculating the gradient in-place. The following is not allowed if you want to use a variable for calculating a gradient:  

a = torch.zeros(1)  
use a for calculation  
a[0] = torch.zeros(1)  
or  
a += torch.ones(1)  

**27 July 2019**  

Added a function that calculates accuracy of a RNN-batch. It still doesn't take into consideration the padded characters that becomes removed in the end of a batch.  

One other thing I noticed is that alot of functions have co-dependencies of functions I wrote. This makes it hard for me to cut out the functions I want to change to my current project-file because they don't see the functions I import from files. (which sounds odd, I have to doublecheck this). My current thoughts is to at least separate all functions into a separate file that does not have any dependencies on other functions I've written.  

**28 July 2019**  

Fixed accuracy-function to adapt to padded characters being removed in forward-pass.  

Started moving functions I wrote that do not dependo on other functions I wrote to a separate file.  

Added forward function go SGRU-module which was missing when I tried to generate a sequence from SGRU-model.  

**29 July 2019**  

loss from training is no longer divided by batch-size because the default reduction in nn.NLLLoss() is mean which means the loss is already divided by the batch-size. The question is if I should stop dividing by the number of character processed as well?  

Added collection of a moving average of the training-loss to the StatsCallback function and a debiasing term.  

SGRU seems to be working properly now.  

Goal for this week is to visualize the max and min gradient(average?) when training the network to see if exploding gradient is a problem in my setup. And introduce gradient clipping with hooks.  

I train a SGRU (3 GRU layers) and compare to a RNN with multiple intrinsic layers (around 5). Training for 30.000 iterations they have similar validation accuaracy. I would have thought that the SGRU would outperform the RNN much more at this point in training. I haven't however controlled my training for exploading gradients (I do a sequence length of 30) and I haven't used momentum, weight-decay or dropout in my models which I think could change the outcome. Experiment saved as experiment 2.  

**30 July 2019**  

So I was trying to find a way to print the gradients being calculated at each time-step which doesn't seem to be possible using the current pytorch architecture. All I can do is print the final gradient calculated for my module.  

Not to be fair, we will see if the summed up gradient is exploding or not but clipping the gradient during the gradient calculation is what I really want to do and I think it would be a better solution which would give better results. And it would be possible to see the limits of how long sequences I can train if it is possible to visualize how the gradient is propagated backwards through the RNN when calculating the gradient...  

**31 July 2019**  

I was looking into hooks and I'm thinking about a good way to store the values from the gradients. Apparently pytorch has a forward-hook as well as a backwards-hook. The backwards hook takes what I believe is the input to the gradient calculation in each layer as well as the output (the produced gradient). Maybe the forward-hook makes it possible to capture the values calculated in the layers on the forward-pass.  

I also did some thinking on why it might not make sense to clip the gradient during the calculations. Now, calculating the gradient in a backwardspass my original thoughts were to change the gradient as it is being calculated by clipping it early on or midway through out the calculations to avoid one of the last (in the backwards-pass) layers to blow up.  

If you do that, it doesn't makes sense to clip of the top of a gradient (only remove parts of the values that reach over a certain threshhold) because it is a nonlinear operation and it could cause gradient-calculation errors that propagate to the later stages of gradient calculations.  

Now if you make a linear-clipping operation that only changes the magnitude of the produced gradient in the later stages/layers we shouldn't lose any information.  

So what I should do is to first look for a paper on gradient clipping and get some idea of what state of the art advanced gradient clipping is.  

Also, if we can print the forwardpass activations maybe we can clip the gradient in all timesteps by changing the activations of each timestep in the forwards-pass.  

So, plan is: read paper; plot the gradient by using hooks and experiment with longest sequence before gradient blowup or collapse.  

**1 August 2019**  

I wrote HookCallback to save the mean and max gradient from training my RNN networks. Plotting the log of these functions and using a sequence length of 100 chars didn't seem to give a problem with vanishing gradients. The lowest value observed was around e-15. Training with a vanilla RNN and a GRU module, the RNNs gradients were less stable than the GRUs of which all the layers gradients were straight lines. The RNN did have 16 layers and the GRU had 6 layers so the unstabilities are expected from the RNN. The GRU however is using sigmoids and therefor is at larger risk of the vanishing gradient problem but the GRU also has the update-gate which should aleviate the vanishing gradient problem and the use of Sigmoid functions.  

I tried to train with and without momentum. Results can be seen in exps/3/. Training without momentum with sql=100 is not a problem. But adding momentum to the RNN drives the gradients to zero. Thats why the plots are without the logarithm in when using momentum. The GRU however manages to start training and get away from collapsing gradients. Maybe the momentum in the case of RNN can be eased in from a value of 1 (or more) to a lower value using RMSProp so that the training can take of first.  

Anyway, I don't plan to use RNN in the future, but GRU, LSTM or Attention. The above experiment I belive is a proof that the GRUs gates can bring the model out of a state of low gradients.  

I am thinking about what the lowest maximum value of the gradients a network can tolerate without stoping to learn. The momentum should also help with the vanishing gradient problem. I haven't added a measurement of the minimum gradient, only the max and the mean. I will add this parameter value as well now.  

What I should do now is to build a LSTM, make experiments with stacked LSTMs and see which of these four (GRU, SGRU, LSTM, SLSTM) works best.  

Put the functions I wrote now into a module for later use.  

Then move on to building my first attention model.  

**2 August 2019**  

So, I realized that I have another option when building both the LSTM and the GRU models. When merging the input and the hidden-state in GRU I used elementwise summation. I could also and I'm in favour of concatenating the vectors instead of pes. So, I will try to do this with both the GRU and the LSTM. In the LSTM, the gates can optionally also merge the old cell-state.  

I realized that I split up the hidden-state and the input in the GRU by using a linear-layer. This won't be necessary if I concatenate the input to the gate instead of piecewise summing them.  

I started writing the LSTM-module. Instead of adding another input/output (the cell-state) I will create a tuple holding both the hidden-state and the cell-state and feed that in the position of the hidden input/output for the fit_rnn methods I already wrote. The only difference is now that the tuple/hidden input is handled differently by the LSTM-modules functions.  

Actually thinking more about this, I really feel the need to do an experiment to see how PyTorch handles these different scenarios – and do some calculations on a paper. I can't solve it in my head, how does splitting up and feeding the same input to different linear-layers impact the backwards-pass. And, what is the upside/downside of concatenating opposed to elementwise addition of two inputs?  

**4 August 2019**  

So I was thinking a little bit more about this. In the case of my GRU where x and hidden input are added together before being feed to a linear layer. I guess one advantage could be that splitting an input with a linear layer could be to reduce the problem of vanishing gradient? At least if you don't feed x or hidden through a linear layer before adding them together.  

What is the initial average output from a linear layer?  

I get the feeling but I cannot back it up by an argument that concatenating two inputs is better than adding them together. You introduce a co-dependence between the two inputs that I believe is harder to learn. (imagine trying to turn down the impact of one variable e.g x_1 and loosing the information brought to it by h_1). This can be elievated by the structure of (i) below. But using the structure (iii) could bring about the same result without introducing codependencies and using fewer weights. 

In my GRU e.g this is what happens:  
(i)   W_21 * (W_11 * x + W_12 * hidden)  
consider the other possibilities:  
(ii)  W_21 * (x + hidden)  
(iii) W_21 * (x _concatenate_ hidden)  
(iv)  W_21 * (W_11 * x _concatenate_ W_12 * hidden)  

I haven't read any articles analysing and comparing these structures yet. Another possible structure would be elementwise multiplication of elements which is talked about in this article: https://medium.com/octavian-ai/incorporating-element-wise-multiplication-can-out-perform-dense-layers-in-neural-networks-c2d807f9fdc2

(v)   W_21 * (x * hidden)  

Finished building my LSTM-module. This version concatenates the inputs to the gates. It also takes the cell-state as input to input-gate and forget-gate in addition to the x and the hidden-state.  

I changed the unpad function to conditionally take a cell as input and return the changed extra cell-element. I don't remember why I put the if statement: if len(hidden.shape) > 2: ... hidden[:,idx] into that function but hopefully it will work by cutting up the cell[:,idx] in the same way.   

**5 August 2019**  

Added a linear-layer before the softmax in my LSTM-module and it now works properly.  

Started writing an Attention Module which uses GRU-RNN. I'm building it from the paper that introduced the attention paper. My guess is that the decoder-RNN is not the same as the encoder-RNN in the paper but the decoder takes as input the hidden-state output from the encoder-RNN.  

A weight-decay of 0.005 is to much for my GRU and LSTM modules. I will try a lower value in the future but atm I'm not using weight-decay when comparing performance between GRU and LSTM.  

**6 August 2019**  

Found a webpage that proposes to initialize the bias of the forget-gate to 1. For some reason it prohibits the gradient from decaying fast. [http://mlexplained.com/2019/02/15/building-an-lstm-from-scratch-in-pytorch-lstms-in-depth-part-1/]  


**7 August 2019**  

I sidetrack a littlebit and start reading the MoGlov paper based on the Glow model. I get stuck on the actnorm layer which is described as a affine transformation.  This is a linear transformation that is invertible. It's unclear for me why this layer is relevant for the flow algorithm to work. It would be nice to know why this is choosen as one part of three layers in the Glow model.  
$a_{t,n} = s_{n} \odot z_{t,n} + t_{n}$  


**9 August 2019**  
Spent even more time reading the Glow paper [https://arxiv.org/pdf/1807.03039.pdf] and the Affine coupling layer paper [https://arxiv.org/pdf/1410.8516.pdf].  

Generative models are used for non-deterministic synthesizing of videos, images text etc. They learn the distribution(s) of the input data. The learnt distributions can be generated by reversing the networks flow.  

This is why the moglow model generates a realistic representation of character-movement. It has learnt the distribution of different natural movements. Mixes of these motions are sampled from the model and combined in a believable way.  

The glow paper mentions that flow-based models allows interpolation between datapoints. It is unclear to me if the interpolation of the distribution of two different character-movements would be presented in a believable manner.  

The question I'm asking is if the interpolation of two distributions is a distributions which during presentation can not be discerned as belonging to a different category than the two original distributions.  

If so the distributions must create a continous subspace of categorical movements. I'm really interrested in this property of the flow networks (IF what I conjecture here is true).  

**21 August 2019**  

09:30 -> 13:00  

Back to Attention. I finished the mockup of the Attention model. It is based on the first paper about attention and I implemented the global-attention model. In this model I couldn't find an equation for the c_t (the contextvector) so I assume that the global align vector a_t equals c_t in this model.  

Furthermore it is still unclear to me if I should always innitiate the Attention model with a zero-hidden-state or if I should use the last output hidden state from the previous training.  

There are going to be multiple problems with the setup that I'm using right now if I want to train prediction of tweets with the raw Attention model I built.  

(i) My attention model asserts a fixed sequence length.  

(ii) If I train with batches the hidden states are going to shrink in batch-size(!) at some point in the process of training the Attention module. Actually, this doesn't have to be a problem now that I think about it.  

problem (i) could I think could maybe be handled by a break statement.  

14:00 -> 16:00  

So I started training and I just found out the CUDA device-side triggered error which is due to the shape of the h_tilde I believe being the wrong shape. I will look at the shape issue next session.  

**22 August 2019**  

11:00 -> 14:10  

So I fixed the size problem. There was an other problem because I do in-place operations on my tensors that are used for gradient calculations. This is not permitted. I fixed it by replacing torch.zeroes() tensors with lists. When I think about this solution I believe the problem can be solved by changing torch.zeros() to torch.tensor() because the latter doesn't assign any values to the memory.  

Another problem surfaced when I try to train attention with batches larger than 1. This error is weird because it seems to me that somewhere after the padded elements (value 0) non padded elements returns. This can't possibly be because if that was true this error would have surfaced when training my other modules.  

However thanks to me implementing callbacks I can easily print out the training data before running each batch_forward. This is what I'm going to do.  

15:00 -> 

So the problem is that I don't save the last hidden state for all linel of the batch. This happens when some lines stop earlier than others. I would need to solve this by looking at each line in the hbar variable and choose the last element from each line. At the moment I only choose the absolute last element which grabs only one hidden state from the hbar variable.  

I don't know how this last part of the batch affects the training. I don't think it should be a problem that some attention updates are not the full sequence length.  

**23 August 2019**  

13:30 -> 

So there are two main problems when training Attention with batch-size > 1.  
(i)  In the end of one batch, different lines in the batch will end earlier than others. Now because we feed the last hidden states from the previous batch to the next batch, we need to save the end hidden states from each line when(!) they end, not only take the hidden state from the last line we are training with (which will usually only be one hidden state (if not multiple – in this case tweets have the same length and thus ends at the same character.  
(ii)  How are we to define the alignment function? How do we align the h_t with the other hidden states (hbar_s)? when they have different sizes?  

I decided to preserve the output from the alignment method having the same size as h_t. because h_tilde will consist of h_t and c_t and they have to have the same size (c_t depends on the size of h_t).  

Two cases: 
(1) bs h_t less than hbar_s: shrink hbar_s because we don't want to calculate alignment between a tweet character that does not exists in h_t.  
(2) bs h_t larger than hbar_s:  
(2,1) add zeros to hbar_s to fill in the void that is multiplied with h_t.  
(2,2) we could also add the zeros to the output of the score function representing the void.  

I choose to do (2,1).  

I did solve problem (ii) and I'm about to solve problem (i).  

So, I'm a little bit annoyed that I didn't realise earlier that the self-attention used by Google ditched the RNN implementation that was propsed in the original paper. I remember that I thought it was odd that they were using RNNs because it said in the Attention is all you need paper that the transformer could be trained in parallell.  

Anyway. I think I will implement the transformers self-attention mechanism instead.  

**26 August 2019**  

10:00 -> 13:00  

In the instructions for the torch softmax function it says that the given dim (2) will be the dimension that will sum to 1. This is true because if you send in a matrix x.shape = [1,3,3] if we choose an element from the first dimension, it will sum to 1 (torch.sum(w.squeeze()[0] = 1). 

Compare this to (eq 2) w = x.squeeze().exp()/torch.sum(x.squeeze().exp(),0). torch.sum(w[0]) = 1.  

e.g a.shape = [3,3]  

At first I didn't realise that the element we choose is the vector that we will sum up and divide by. I thought that the choosen dimension in torch.sum was the dimension that would be keept unchanged. When in reality if I choose the row (dimension 0), the summation will be carried out over the columns and I will be given back a row that consists of elements which are the sums of the columns in each row position.  

Now because of how I wrote the (eq 2) because the sum will be the same nomatter which dimension I sum over because x in this case is equal to its transpose. And when you elementwise divide a [3,3] matrix with a vector of dimension [3] it will always expand the vector in the column dimension (which is missing). This results in the a matrix that always sums to 1 if you hold the columns and sum over the rows.  

These basic things mindfuck me over and over again. I feel like I have built a conception of how things work and I don't really digg down to the basics and I redo the same mistake over and over again. Instead I should be thorough and always find the reason to my missconception.  

So I'm following this blog http://www.peterbloem.nl/blog/transformers that explaines and builds a transformer from scratch using torch-tensors and torch-operations.  

Everything seems reasonable at this point. I understand the attention mechanism without learnable parameters. Now I try to add learnable parameters which are represented by the query, key and value weigths. Also notice that none of these weights include a bias. (It might have something to do with the matrix being operated on is symmetric but I haven't thought to much about it yet.)  

14:15 -> 15:30  

The reason we use view instead of reshape is because view always uses the same memory space. I.e we don't know if or when reshape() makes a copy so we should use view. Also when you want a copy, use clone()...  

The same memory ability holds for transpose(), narrow(), expand()  

I realized a new thing about the nn.Linear class. It will always operate on the last element in the tensor put into it. Normally we have size [batchsize, xsize] but right now I have [batchsize, time, xsize]. When we feed that vector to nn.Linear(xsize, nheads*xsize) it will still only operate on the last element which is xsize.  This means I don't have to or shouldn't flatten the input vector before.  

One concept that I had trouble understanding is the mask. The mask is just a torch tensor with 0's and 1's. That is elementwise-multiplied with the input (e.g input = [b,t,k,k]. input * mask.rep(b,t,1). The mask has 1's in the bottomleft and 0's in the topright corner. Multiplying by the mask will make it impossible for the timesteps in the beginning to use information from the future timesteps because they will be set to 0's.  

Wrote the mask function, the one I copied seemed to use the wrong dimensions.  

Finished the encoder block of the google transformer, will try to finish the decoder block next session.  


**27 August 2019**  

14:00 - 15:30 

In the GoogleTransformer they seem to merge the output from the encoder with the output from one selfattention block on the target (Y) – feeding it to another self-attention block. How this merge happens is unclear for me.  

Do we feed the encoder output (X) to some of the heads and the decoder output (Y) to the other heads of the self-attention block? and use this blocks result as a prediction?  

I'm going to look around some more code that directly implemented this paper. This is the only part of the Transformer that I don't understand thoroughly.  

Wrote a feedforward-block for the GoogleTransformer, it consists of two linear-layers with a relu between them.  


**29 August 2019**  

Looking closer at the paper and the images is it clear that the Value and Key input to the second "self-attention" block comes from the decoder output and the Query comes from the first self-attention block in the decoder block. This is mimicing the setup of the original attention paper based on RNNs. The target (Y) is there compared to the input (X) to create the alignment vectors (in the Google Self-Attention Y corresponds to Query and X corresponds to Keys. But the google Self-Attention adds another step after the softmaxing the result of the Query and Key before feeding the result to a linear layer. It compares the input (X) again to the probability matrix (in lack of better wording – the output from the Query/Key) by matrix multiplication.  

This makes sense, it doesn't make sense to do the opposite procedure and use Y as the Value. Y is the target that we are going to compare the output to, why would we train the network on the output and then compare it to the output again? The information is in the input.  

So I rewrote the self-attention functin to take Query, Key and Value as parameters - optionaly a mask.  

A question comes to my mind, if we are going to predict say the next character in a sequence and we use the google transformer. the input when training X is shifted one step to the right producing Y. But when we predicting we are missing the last character as input (in Y). How does the prediction come in?  

In a selfattention RNN, we can always take the hidden state that we produce in order. But not using RNNs, the size is fixed for the self-attention mechanism in Googles transformers.  

Straight from the top of my mind I would guess that we shift the sequence one step to the right and encode the last character in the sequence as a blank state. You would then have basically two different Y vectors. One that contains the original right-shifted vector that we use as target comparing with the output of the transformer, and one Y vector with the last element changed to a blank state.  

Another possibility is that the mask is created in a way such that it covers out the last element such that it is not considered at all as an input. 

Questions, are we waisting resources by making a prediction for all the characters in the target-vector? Most likely not, this information has relative positional information that we need to learn sequential information... 

I understand that we can use the transformer for multiple purposes, not only sequence prediction but also marking e.g the most influencal parts of a sequence which then could be used as input to solve another problem. I will read the paper Attention is all you need again to see if I missed something.  

23:40 ->  

I take back what I wrote about using the mask to hide information about the last element in Y. The google transformer uses a Residual connection that passes Y besides the masked self-attention layer. So masking Y doesn't solve the last element prediction problem.  

**30 August 2019**  

Ok, so it seems that the Transformer presented in the googles paper "Attention is all..." is used solely for language translation. So it is not built to handle character prediction in sequences.  

It is still unclear to me why it says in the paper that the input to the decoder is "shifted right". 

When doing sequence next character prediction maybe I should just skip the decoder block. Question is if I should use the mask when training a transformer that predicts next character in a sequence model.  

Still, I don't understand how the Transformer does translation after training. When making translations, we can't feed the decoder with a target sentence. So how does it work when we only have the input X (e.g english sentence to be translated to German)?  

So I found the answer to how you use the Transformer when translating after training. We initiate the output as zeros with the first row indexed as the <sos> character. We run the Encoder once and then repeatedly run the decoder phase generating a new character each time until we reach the <eos> character. Explanation can be found at the bottom of this article https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec.  

So I guess this is what they mean with shifting right. When we filled the outputvector with characters, not hitting the <eos> character we shift the output once to the right and generate the next character.  
  
This poses another question, when should we update the encoder that holds the context and some information for translation? Should we shift it right as well when we filled up the output vector? Should we compute the next chunk of the sentence with the encoder? and wait until we generated a number of characters equal to the sentence length and then put the third chunk of the sentence to the encoder etc...?  

Reading the Transformer-XL (Extra Long) paper and it seems to adress these questions as well as long-term historical dependencies.  

**31 August 2019**  

In the Transformer-XL paper Values and Key-weights are multiplied with a concatenation of the previous and current input-sequece. This creates a segment-level recurrence in the hidden states.  

However the paper propses to cache the old hidden-states to be reused during evaluation and refer to them as the memory $m \part \mathbb{R}^{M x d}$. 

M is equal to sequence length during training but increased multiple times during evaluation. It is unclear to me how these cached hidden states are used during evaluation. This is the only thing that I don't understand about the Transformer-XL paper. https://arxiv.org/pdf/1901.02860.pdf  

I finished reading the Transformer-XL paper. My plan now before I implement the Transformer-XL is to finish the Vanilla-Transformer. I'm only missing the positional-embedding matrix functionality.  



