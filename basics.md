
[blog GPT2]: <http://jalammar.github.io/illustrated-gpt2/>

[paper flow]: <https://openreview.net/pdf?id=ByftGnR9KX>

[github word2vec]: <https://github.com/bollu/bollu.github.io#everything-you-know-about-word2vec-is-wrong>
[medium adagrad]: <https://medium.com/konvergen/an-introduction-to-adagrad-f130ae871827>
[medium rprop]: <https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a>
[research esgd]: <https://www.researchgate.net/publication/272423025_RMSProp_and_equilibrated_adaptive_learning_rates_for_non-convex_optimization>
[paper eve]: <https://arxiv.org/pdf/1611.01505.pdf>
[youtube eve]: <https://www.youtube.com/watch?v=nBE_ClJzYEM>
[medium LSTM 1]: <https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714> 
[medium LSTM 2]: <https://medium.com/datadriveninvestor/recurrent-neural-network-rnn-52dd4f01b7e8> 
[GRU]: <https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be>
[wiki LSTM]: <https://en.wikipedia.org/wiki/Long_short-term_memory>
[paper XLNet]: <https://arxiv.org/pdf/1906.08237.pdf>
[paper XLT]: <https://arxiv.org/pdf/1901.02860.pdf>

[paper layernorm]: <https://arxiv.org/pdf/1607.06450.pdf>
[medium skipthought]: <https://medium.com/@sanyamagarwal/my-thoughts-on-skip-thoughts-a3e773605efa>
[blog nonzerois]: <https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html>
[tips on training RNN]: <https://danijar.com/tips-for-training-recurrent-neural-networks/>

[skymind attention]: <https://skymind.ai/wiki/attention-mechanism-memory-network>
[paper attention1]: <https://arxiv.org/pdf/1508.04025.pdf>
[paper google attention]: <https://arxiv.org/pdf/1706.03762.pdf>
[animated attention]: <https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/>
[medium all you need attention]: <https://medium.com/@Alibaba_Cloud/self-attention-mechanisms-in-natural-language-processing-9f28315ff905>

[article QAmodel]: <https://towardsdatascience.com/nlp-building-a-question-answering-model-ed0529a68c54>

[tensor basics]: <https://deeplizard.com/learn/video/fCVuiW9AFzY> 
[medium BGRU]: <https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66>  
[medium ULMFIT]: <https://medium.com/mlreview/understanding-building-blocks-of-ulmfit-818d3775325b>  

[youtube ELMo]: <https://www.youtube.com/watch?v=9JfGxKkmBc0>

[generative models]: <https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html>
[VAE article]: <https://jaan.io/what-is-variational-autoencoder-vae-tutorial/>
[affine coupling]: <https://arxiv.org/pdf/1410.8516.pdf> 

[xlexp XLNet]: <https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained/>

[article GCM]: <https://towardsdatascience.com/how-to-build-a-gated-convolutional-neural-network-gcnn-for-natural-language-processing-nlp-5ba3ee730bfb>  

[article PN]: <https://arxiv.org/pdf/1506.03134.pdf>
[blog transformer]: <http://www.peterbloem.nl/blog/transformers>

[blog wordembeddings]: <https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html>  

[medium seq2seq attention]: <https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53>




## Concepts to look closer into:  

- *ELMo*  

## Papers to look closer into:  
Jeremy Howard Recommends:  
- *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* 
section 2.2  
- *Understanding the difficulty of training deep feedforward neural networks*  
- *All you need is a good Init*  
- *Fixup Initialization*  
- *Self-Normalizing Neural Networks*  
- *Bag of Tricks for Image Classification with Convolutional Neural Networks - 2018 dec*  
- *LAMB optimizer paper (don't know what the name of it is*  
- *mixup: Beyond Empirical Risk Minimization*  

## Things I don't know where to put yet:  

- **combine categorical and continuous variables in a network:** https://datascience.stackexchange.com/questions/29634/how-to-combine-categorical-and-continuous-input-features-for-neural-network-trai  

- **about floating point numbers https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html**  

- **checking gradient and more http://cs231n.github.io/neural-networks-3/**  

- **Time-Series decomposition into Trend and Seasonality https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/**  

- **What is ensemble learning? One paper proposes it reduces variance in the trained model**  

- **https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network**  

- **[tips on training RNN]**:  

- **Skip Thought Vectors**:  [medium skipthought]  

- **Non-Zero initial state**:  [blog nonzerois]  
Train the inital-state as a model parameter and/or use a noisy initial state.  

- **Layer Normalization**: Batch-Normalization for RNNs [paper layernorm]  

- **PyTorch Tensor Basics**: [tensor basics]

https://medium.com/@kolloldas/building-the-mighty-transformer-for-sequence-tagging-in-pytorch-part-i-a1815655cd8

## Uncategorizable Concepts:  

- Temporal Convolution  ([article GCM])  
Convolving only on the left-hand side of the sequence as to not leak information from the future.  

- Gated Linear Units (GLU)  ([article GCM])  
$A = X\*W + b$  
$B = X\*V + c$  
$A \otimes sigmoid(B)$  

- Pointer Networks ([article PN])  
seq2seq network that uses "Content Based Input Attention"  
$u^{i} = v^{T} tanh(W_{1}E + W_{2}d_{i})$  
$a^{i} = softmax(u^{i})$  
there are two ways to use the CBIA:  
**i)**  
$\hat{d_{i}} = a^{i} E$  
$[\hat{d_{i}}; d_{i}]$ is feed to a linear-layer -> $d_{i+1}$; $d_{i+1}$ is feed to a linear-layer -> prediction  
**ii)**  
use $a^{i}$ to decide which encoder hidden-state to use as input to decoder.  
$E(max(a^{i})) = e^{j} = d_{i+1}$  
problem is that one of the elements might point back to itself.  

## NLP: Natural Language Processing  

NLP generally requires multiple steps of pretraining the input- and output-data. 

- (pretraining) **AR**: AutoRegressive language modelling  
seeks to estimate the probability distribution of a text corpus using a autoregressive model either as a backwards- or forward-product.  
Problem:  encodes only uni-directional context.  

I don't understand the what this method is doing, we teach the model what to expect given a sequence starting from the left or the right. 

- (pretraining) **AE** AutoEncoding **BERT**: Bidirectional Encoder Representation from Transformers  
given an input token sequence with a portion of tokens replaced by a mask – pre-train network to recover original tokens from the corrupted version.  
Problem:  not able to model the joint probability using product rule – BERT assumes the independent tokens are independent of each other given the unmasked tokens.  

- **MC machine comprehension (NLU - Natural Language Understanding)**:  
**single turn**:  
(1) question encoding  
(2) context encoding  
(3) reasoning, and finally  
(4) an- swer prediction  


- **Language Model** ([blog GPT2])
A language model is – basically a machine learning model that is able to look at part of a sentence and predict the next word. E.g Word2Vec. 

- **QA model**  ([article QAmodel])  
A regular Question- and Answering-model presented in the article above is built with bidirectional attention. Attention from Context to Questions and from the Questions to the Context.  

- **Gated Convolutional Model**  ([article GCM])  
GCM for language analysis uses Temporal Convolution – model is obstructed from knowing information from the "future".  

- **FLOW (context of MC)**: ([paper FLOW])  
**-** base neural model  
**-** flow mechanism *encodes history*  

- **Summarization**:  
**Extractive**: *sentence classification problem*: created a summary by identifying (and subsequently concatenating) the most important sentences in a document.  
**Abstractive**: *sequence-to-sequence problem*: ; 

- **Pre-Processing**:  
- *Trigram Blocking*:  
Trigram Blocking is used to reduce redundancy. Given selected summary S and a candidate sentence c, we will skip c if there exists a trigram overlapping between c and S. This is similar to the Maximal Marginal Relevance (MMR).  

- (pretraining) **XLNet**:  ([paper XLNet])  ([mlexp XLNet])  
**(i)**  maximizes the expected log likelihood of a sequence w.r.t all possible permutations of the factorization order  
**i.e capturing bidirectinal context**.  
**(ii)** provides a natural way to use the product rule for factorizing the joint probability of the predicted tokens  
**i.e no independence assumption**.  
**(iii)** integrates the segment recurrence mechanism and relative encoding scheme of Transformer-XL with adjustments for the arbitrary factorization order and target ambiguity ((i))  
**i.e improved performance for longer text sequences**  

- **ELMo**: [youtube ELMo]  
**i)** Use multiple layers of recurrent units in the encoder  
**ii)** Keep all the internal layer representations, in addition to the final recurrent layer.  
**iii)** For any downstream task, create the task-specific embeddings as a linear combination of all the internal layer representation.  
As I understand it, ELMo is a stacked bi-directional RNN where the output for each stack-layer (LSTM/GRU) is run through a softmax and then added together with the output from all the other stack-layers + the input (goes through softmax) and then scaled by a trainable gamma parameter before (not sure about this) being feed to a final soft-max layer and a prediction.  

**Embeddings**: [[blog wordembeddings]]

- **Word Vectors**: https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/  

- **Skip-Gram**: reverse of CBOW  

- **CBOW**: reverse of Skip-Gram  


## Generative Models:  

Generative Models try to solve the problem which DL-models have that they work porly with data that has been taken from a different distribution than the training data. [generative models]  

Likelihood methods can be divided into three sub-categories:  
(i) **Autoregressive Models** (e.g GANs)  
uses discriminator to tune the models distribution and minimize the difference from the target distribution.  
(ii) **Variational Auto-Encoders**  
uses non-deterministic encoders to learn the target distribution. Pros, can be trained with parallelization.  
Allows interpolation of datapoints e.g in synthesized data.  
(iii) **Flow-Based generative models** (e.g GLOW)  
uses chains of deterministic transformation functions to learn the target distribution. 
Also allows interpolation of datapoints e.g in synthesized data.  
Calculation of gradient using invertible functions requires non-exponential memory consumption.  

#### GANs: Generative Adverserial Networks:  
Uses two models  
i) *Discriminator* "discriminates" by guessing if the sample is real or generated.  
ii) *Generator* the generator tries to fool the discrimator by generating convincing samples.  

#### VAE: Variational Auto-Encoders  

Structure of VAE is an encoder and a decoder and a loss-function.  [VAE article]  
Loss-function uses a regularizing term: the Kullback-Leibler divergence (relative entropy) – how much does one propability distribution differ from another?  Because p (distribution of decoder) is a normal-distribution with mean 0 and variance 1 it forces the decoders distribution to have the same shape.  

#### NF: Normalizing Flows:  

The reason we use Normalizing Flows is to transform a gaussian distribution to a more complex distribution that better fits the function and data that we want to model. The method of transforming g-d in multiple steps by using multiple functions is called a Normalizing Flow.

Each of these transform-functions should have two properties:
(i)  easily invertable
(ii) it's Jacobian is easy to calculate 

GANs using encoder/decoder that produces a invertible differentiable nonlinear transformation. As such the dimensionality of the input and the output is the same. And chaining these invertible functions $x = f(z) = f_{1}(f_{2}(...f_{N}(z)))$ into a structure that is called normalizing flows.  

Glow: three steps:  
(i) Actnorm (variation of batch-norm in the context of flows)  
$a_{t,n} = s_{n} \odot z_{t,n} + t_{n}$  
It's an affine transformation – retains relative distances and angles between points in the transformed image. Affine maps do not have to preserve the zero point => all linear transformations are affine but not all afine transformations are linear.  
(ii) Linear-layer  (standard linear layer)  
(iii) Affine coupling  [[affine coupling]]  
As I understand it, this is a way to learn the distribution of a set of inputs, it is still unclear why this is better or different than using another actnorm layer which also is a trainable affine function.  
For affine coupling the input is divided into two parts $x_{1}$ and $x_{2}$. $x_{1}$ deterministically generates the parameters of an affine-function $f_{1}$. $x_{2}$ is feed to the affine transformation-function $f_{1}(x_{2}) = z_{2}$
The output from the layer is the combination of $x_{1}$ and $z_{2}$, $x_{1}$ is needed to calculate the inverse of the affine coupling.  


## SGD – Statistical Gradient Descent:
Hessian Free Optimization: [Martens, 2010]
Use second order information from the second derivation to update weights.
Newtons method: Loss function is locally approximated by the quadratic
expression containing the Hessian – the second derivative matrix.
Calculating only some elements of the Hessian is possible, not all.
Choose the elements with information about the the direction of the Gradient Descent.

- **AdaGrad**:
Algorithm adaptive scaled learning-rate for each dimension. ([medium adagrad])  
AdaGrad is almost the same as SGD but each individual parameter update is scaled (divided) with the accumulated squared gradient from every previous gradient computation. In the long run this term goes to 0 for all parameters.  

- **RProp**:  
RProp doesn't work with mini-batches because it uses the sign of the gradient.  
Goal is to solve problem with gradients varying a lot in magnitudes. ([medium rprop])  
Rprop combines two ideas:  
**(i)**  only using the sign of the gradient  
**(ii)** adapting the step size individually for each weight.  
Rprop looks at the sign of the two(!) previous gradient steps and adjust the stepsize accordingly (intensify or decrease).
It is also adviced to limit the stepsize between a minimum and maximum value. 

- **RMSProp**:  
RMSProp does work with mini-batches by using moving average of the squared gradient for each weight. ([medium rprop])  
<br/> This still doesn't resemble the RPROP algorithm. The RPROP algorithm decreases the learning-rate if we go the other direction, and increase it if we are going the same direction.  
In RMSProp the algorithm doesn't remember which direction it went in the previous iterations. It only matters what the magnitude is of the gradient.  
Leslis N. SMith says it's a biased estimate, I believe it is in relation to the Hessian. RMSProp comes close to the Hessian. (Whatever that means?)  

- **ESGD**:
Equilibrated SGD – unbiased version of RMSProp. ([research esgd])  <br/>
In the paper the authors proposes an update of the moving-average(?) every 20th iteration because it would have the same calculation overhead as RMSPROP. And still (I guess) better performance. I haven't found any good source of the implementation of the paper yet.  

- **ADAM**: – ADAptiv Momentum estimation – RMSprop + Momentum. ([youtube eve])  
<br/> $$\theta_{t+1} = \theta_{t} - lr \frac{m_{t}}{\sqrt{v_t} + \epsilon} $$  
Adam takes small steps in steep terrain and large steps in flat terrain.  
$m_{t} = \beta_{1} m_{t-1} + (1-\beta_{1}) grad_{t}$ – average recent gradient  
$v_{t} = \beta_{2} v_{t-1} + (1-\beta_{2}) {grad_{t}}^{2}$ – average recent deviation in the gradient  
$v_{t}$ is related to the second derivative and is in general close to constant. 

- **EVE**: – evolution of Adam () – locally and globaly adaptive learning-rate ([paper eve], [youtube eve])  
<br/> $$\theta_{t+1} = \theta_{t} - \frac{lr}{d_{t}} \frac{m_{t}}{\sqrt{v_t} + \epsilon}$$
$d_{t}$ is the only difference between Adam and Eve, has two objectives:  
**(i)** large variation in the Loss-function between steps should be given less weight -> take smaller steps.  
**(ii)** are we far from the minium (L*)? -> take larger steps.  
<br/> $$\frac{1}{d_{t}} \propto \frac{L_{t} - L^{\*}}{|L_{t} - L_{t-1}|}$$  
<br/> problem (**ii**) Stepping away from $L^{\*}$ might incrementally take larger and larger steps and blowing up.  
solution (**ii**) $\frac{1}{c} \leq \frac{1}{d_{t}} \leq c$  
Also add smoothness to $d_{t}$ with another running average ($beta_{3}$).  
Use Adam to estimate $L^{\*}$ or set $L^{\*} = 0$  

- **CLR** – Cyclical Learning Rate:
Requires almost no additional computation 
Linear change of learning rate easiest to implement (my thoughts), chosen because any nonlinear change between largest and smallest gave the same result.  

## RNNs: Recurrent Neural Networks
Used to learn periodical patterns from data.  Vanilla RNNs functions as a hidden-markov-model.  
Problems: vanishing- and exploding-gradient.  Exploding gradient problem can be solved by "clipping" (setting upper and lower limit for) the gradient. The vanishing-gradient problem is harder to solve.  
parts:  
$x$ - data  
$o$ - output  
$h$ - hidden-unit  

- **LSTM**: – Long Short Term Memory  [medium LSTM 1]  
Stronger than GRUs, can easily perform unbounded counting (don't know what that entails) [wiki LSTM] Both LSTM and GRU are different from Vanilla RNN which replaces the old states with a new calculated one – while LSTM/GRU saves part of the old state and adds the new state on top of it. Both GRU and LSTM(not sure about this yet) can learn patterns that RNNs cannot learn. But neither can be trained using parallelisation.  
LSTM uses a cell-state that is updated without multiplication of Weights and hence can carry information far backwards without being affected by the vanishing gradient problem.  
**gates** in RNNs are outputs from sigma-functions, the gate-gate in the LSTM is called a gate because it's a tanh multiplied with a sigma-function. It decides how much to write to the new cell-state.   
**3 inputs; 2 output states:**  [medium LSTM 2]  
$x$ – input  
$c$ – cell-state  
$h$ – hidden-state  
**4 gates to operate:**  
$i$ – input gate  
$f$ – forget gate  
$o$ – output gate  
$g$ – gate gate  

- **GRU**: – Gated Recurrent Unit [GRU]  
GRU is more computational efficient and almost on par with performance of LSTM.  GRU counters the vanishing gradient problem similar to LSTM.  
**2 inputs; 1 output state**:  
$x$ – input  
$h$ - hidden-state  
**2 gates to operate:**  
$z$ – update gate  
$r$ – reset gate  

- **Bidirectional GRU/LSTM**: [[medium BGRU]]  
As I understand creating a bidirectional-gru/lstm is merging the result of two separate gru/lstms. IIuc (if I understand correctly) the order of the output from the counter-directional rnn/lstm is reversed and then merged in some way, summed, concatenated or elementwise-averaged before being feed to a linear-layer.  

- **Stacked GRU/LSTM**:  
Stacking multiple GRU/LSTM on each other. The output from the first layer of GRU/LSTM is the input to the second layer. The output from the final layer is feed to a linear-layer.  

- **Seq2seq with Global Attention**: [[medium seq2seq attention]]  
Doing sequence to sequence transformation using RNNs as encoder and RNNs as decoder with global attention between. The encoder usually uses multiple layers (not more than 2-3) and bi-directional LSTM/GRUs.  

- **ULMFIT**: - [[medium ULMFIT]]  
Haven't read the whol article. ULMFIT is preprocessing text, then training with a custom dropout for embeddings and hidden layers in RNNs which need to be zeroed out in a different way than weights in a linear-layer.  

## Attention: – [[skymind attention]] [[animated attention]]

The original Attention model propsed in [paper attention] implemented RNNs for the encoder and decoder. This changed when google presented their paper "Attention is all you need".  

*global-attention*: all encoder hidden states are processed by the attention-decoder.  
*local-attention*: a subset of the encoder hidden states are processed by the attention-decoder.  

- **Original attention**:  [[paper attention1]]
 
RNN with encoder/decoder. The decoder is where the attention happens. The encoder hidden-/source-states are saved for processing by the attention-decoder. The Decoder scores each hidden state on an "attention" basis. Multiplies them by their softmax score and sums them up to => $c_{t}$ the context-vector.  

$h_{s}$ – source state  
$h_{t}$ – current decoder hidden-state  
$a_{t}$ – alignment vector = $softmax((fh_{s},h_{t})))$ where f() can have various definitions but include at least $h_{t}$.  
$c_{t}$ – context vector = $g(a_{t},\bar{h_{s}})$  
$\tilde{h_{t}}$ – attentional vector = $tanh(W_{c}[c_{t};h_{t}])$  

these abreviations are taken from the  which should be one of the first attention papers with good results exploring different implementations of the attention concept.  

calculate the h_bar vector containing the hidden states output from feeding the sequence forward in the attention-encoder-RNN. 
calculate the context vector by combining calculating probability by using the attention-decoder-RNN (might be the same RNN as the encoder) and scoring the hidden output for each time-step with the c_t vector.  

- **Googles self-attention**:  [[paper google attention]]

A matrix that puts two different sequences at adjacent sides of a matrix. This matrix explains the relationship between the parts of the two sequences.  
$Googles-Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$  
Q -  search query  
K -  key  
V -  value   
Intuitively, the query represents what kind of information we are looking for, the keys represent the relevance to the query, and the values represent the actual contents of the input.  [[blog transformer]]

- **Transformers**:  

Avoids recurrence by using attention. Allows parallelisation and faster training. 
Relating signals from input and output positions depend only on "distance" between them.  
Drawback: Distance is a linear dependence – averaging attention-weighted positions –   
Solution: Use Multi-Head Attention.  

The query - Q searches over the keys – K of all e.g words that might supply context for it. "Those keys are related to values that encode more meaning about the key word.  
Any given word can have multiple meanings and relate to other words in different ways, you can have more than one query-key-value complex attached to it. That’s “multi-headed attention.”" 

Transformers use Googles self-attention which means no RNNs. There are multiple architectures of transformers, the one presented in the paper ([paper google attention]) includes a encoder and a decoder. The decoder prevents the model from "looking" into the future by masking out information that is not sequential.  

Generating output from the Google-transformer is done by feeding the decoder a sequence-length zeroed matrix that includes one non-zero element (initially the <bos> character). In each forward pass one additional character is generated and added to the input for the next forward-pass.  

- **XL-Transformer**:  ([paper XLT])

learns dependencies that are 80% longer than RNNs and 450% longer than vanilla-transformers.  

**recurrence memory**  

Transformers do not have memory cells/states that are passed on in a autoregressive fashion. The context in the vanilla-transformer (googles) takes it's historical information from the encoder. The XL-transformer aims to extend the amount of information a transformer can learn by feeding it more information from previous chunks of sequences to various extents.  

This information is feed to the Value- and the Key-weights, not the Query-weights. These weights are dimensionally expanded and multiplied with the previous sequential chunk ($X_{t-1}$) concatenated to the current sequential chunk ($X_{t}$).  

This creates a "segment-level recurrence in the hidden states" which extends the context way beyond two segments back.  

**relative positional embeddings**  

This creates a technical problem with the positional-encoding added to the input. To solve this, calculation of relative position is introduced and hence the calculation of MatMul has to be changed. This is done by separating the Content-Embedding and the Relative-Position embeddings when doing the the MatMul calculation. Two additional learnable terms are added $u$ and $v$. 

#### RNN: preprocessing:  

– **Filtering Cycle Decomposition (FCD)**:  



#### RNN: training nomenclature:  

- **Teacher Forcing:**  feeding the ground-truth (yt, yt+1, ...)to the model during sequential training.  
- **Free Running:** 	feeding the output to the model sequentially. 
- **Professor Forcing:**  using a GAN-discriminator to force hidden-states from teacher-forcing and free-running to be close to each other.  

**Weight Initialization**:  
The goal is to have the variance be the same when progressing through the network. If the 
variance is to small, e.g for the Sigmoid function, the output behaves in a linear fashion around 0. 
On the other hand the variance is to high, we will move to the fringes of the Sigmoid function for 
which the gradient is 0 so the network won't train.  

The values for both Xa- and He-activations are based on the histograms of the
activation distribution of Tanh and ReLU activation functions.  

Xavier initialization assumes activations in x(i) follow a
symmetric distribution which is not true for ReLU activation.
Xavier initizlizes the weights so that the variance for x and y in 
each layer is the same. 	

- **Xavier init** (used for tanh-activation-functions)  
$$\mu = 0$$  
$$\sigma = \frac{1}{\sqrt{n_{in}}}$$  

- **He init** (used for ReLU-activation-functions)  
$$\mu = 0$$  
$$\sigma = \frac{\sqrt{2}}{\sqrt{n_{in}}}$$  

**Regularization**:  
Process of adding information to solve  an ill-posed problem or to prevent overfitting.  

- **Large learning-rate**  => regularizational effect.  
- **Small batch-size** => regularizational effect.    
- **Weight Decay: L1 – Lasso Regression**: $$\lambda \sum{0_{i}}$$  
Computational inefficient on non-sparse cases  
Sparse outputs  
Built in feature-selection  
Shrinks less important feature's coefficients to zero.  
- **Weight Decay: L2 – Ridge Regression**: $$\lambda \sum{{0_{i}}^{2}}$$  
Computational efficient due to having analytical solutions  
Non-sparse outputs  
No feature-selection  

**Vocabulary**:  
Epoch:      passing the whole dataset forward and backward through the model once.
Iteration:  one iteration is passing one batch forward and backward through the model once.
BatchSize:
Cycle:      in cyclic learning one cycle is starting at eta_min -> eta_max -> eta_min and
      ending the training there.
Sparse:     Sparsity refers to that only very few entries in a matrix (or vector) is non-zero.
Loss:
Cost:
Accuracy:
Generalization Error: 

**Activation functions**:  

- **Sigmoid**:
If all inputs are positive, the derivative of the cost-function J for the 
different layers are either all positive or all negative weights layerwise.
I DON'T UNDERSTAND WHY. What is the derivative of dJ/dh? Where h is the
sigmoid function? <- see slide 4 Josephine Sullivan.  

cons: exp() is expensive to compute. outputs are not zero-centered. saturated activation kill the gradient.  

Inverse problem:
  input:  set of observations
  output: causal factors that produced them.
  calculate the causes with the result.


Ill-posed problem:
  a problem that is not well-posed:
    # a solution exists
    # the solution is unique
    # the solutions behaviour changes continously with the initial condition.

  For ill-posed problems. There are no unique solutions. Technically the matrix calculation describing the model might be of lower rank than it's size. The column/row vectors making up the matrix are not linearly-independent. (Multi-collinearity)

  Errors in data or limited precision makes the model suffer from numerical instability. Small errors in the initial data becomes large errors in the answer.

Batch Normalization:
	BN has to be changed during testing. We should not use batch-mean or batch-variance. Instead 
	the layer uses a calculated moving-average-mean and -variance. 

	BatchNormalization can be fine-tuned by changing it's momentum value. 
		high-momentum: high lag, slow learning; good when batch-sizes are small  
		low-momentum:  low lag, fast learning;  good when batch-sizes are large 

	https://medium.com/@ilango100/batchnorm-fine-tune-your-booster-bef9f9493e22	
	https://towardsdatascience.com/intuit-and-implement-batch-normalization-c05480333c5b
  	
	# Improves gradient flow through the network.
  	# Allows higher learning rates.
  	# Acts as regularization.
  	# Reduces the strong dependencies on weight initialization.


**Calculating Backpropagation on a paper:**  
Derivative of a scalar w.r.t a matrix:
r = sum(i,j)(W^2)
dr/dW = ?

because W is a matrix we will have a derivative for each of Ws elements:
i.e we will create the Jacobian of W:

(dr/dW11 dr/dW12 etc)
(dr/dW21 dr/dW22 etc)
(etc

Applying the Jacobian_W to W^2 => sum(i,j)(2W)
Derivative of a scalar w.r.t a vector:
in this case we have to create a Jacobian-Vector. The same as above.

Derivative of a Vector w.r.t another Vector:
In this case the symbol above the denominator in the Jacobian will
not be a scalar anymore. It will change depending on the position.
The vectors have to be of same size otherwise the vectors couldn't be multiplied
in the function we are evaluating.

If the Derivative is of a Matrix w.r.t a Vector my guess is that
we will have som kind of sum along the dimension that is not multiplied
with the Vector in the evaluated function.

e.g: dp/ds size(p) = size(s) = [1xC]

dp/ds = (dp1/ds1 dp2/ds1 ... dpC/ds1)
(dp1/ds2 dp2/ds2 ... dpC/ds2)
...
(dp1/dsC dp2/dsC ... dpC/dsC)

Derivative of a Vector w.r.t a Matrix:
Does not exist consistent definition for "Jacobian" of vector w.r.t matrix.

Solution is: to rewrite Matrix as a vector
     and extend the Vector with the Kronicker Multiplication of identity matrix of size len(x).

this is complicated I know. I am not totaly sure that the identity matrix has to have size len(x)

Convolution:
	Dilated Convolution:
		Convolution filters that takes pixels that are not nearest neighbours.


Papers to read:
	Improving Language Understanding by Generative Pre-Training
	BERT: Tre-training of Deep Bidirectional Transformers for Language Understanding
	Adversarial Examples Are Not Easily Detected: Bypassing Ten 




 
LR range test:
  run model for several epochs over increasing learning rate values.
  max_lr  = lr(maximum accuracy) before accuracy get jagged/rough
  base_lr = lr(minimum accuracy) or 1/3 or 1/4 of max_lr


Loss Function Topology:
  Difficulty in minimizing the loss arrises
  from Saddle Points not  Poor Local Minima [Daupin et al.]

  Saddle points have small gradients that slow the learning process.
