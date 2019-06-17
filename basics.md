[github word2vec]: <https://github.com/bollu/bollu.github.io#everything-you-know-about-word2vec-is-wrong>
[medium adagrad]: <https://medium.com/konvergen/an-introduction-to-adagrad-f130ae871827>
[medium rprop]: <https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a>
[research esgd]: <https://www.researchgate.net/publication/272423025_RMSProp_and_equilibrated_adaptive_learning_rates_for_non-convex_optimization>
[paper eve]: <https://arxiv.org/pdf/1611.01505.pdf>
[youtube eve]: <https://www.youtube.com/watch?v=nBE_ClJzYEM>

##### Statistical Gradient Descent:
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
goal is to solve problem with gradients varying a lot in magnitudes. ([medium rprop])  
Rprop combines two ideas:  
**(i)**  only using the sign of the gradient  
**(ii)** adapting the step size individually for each weight.  
Rprop looks at the sign of the two(!) previous gradient steps and adjust the stepsize accordingly (intensify or decrease).
It is also adviced to limit the stepsize between a minimum and maximum value. 

- **RMSProp**:
RProp doesn't work for mini-batch updates because it uses the sign of the gradient, solved by using moving average of the squared gradient for each weight. ([medium rprop])  
<br/> This still doesn't resemble the RPROP algorithm. The RPROP algorithm decreases the learning-rate if we go the other direction, and increase it if we are going the same direction.  
In RMSProp the algorithm doesn't remember which direction it went in the previous iterations. It only matters what the magnitude is of the gradient.  
Leslis N. SMith says it's a biased estimate, I believe it is in relation to the Hessian. RMSProp comes close to the Hessian. (Whatever that means?)  

- **ESGD**:
Equilibrated SGD – unbiased version of RMSProp. ([research esgd])  <br/>
In the paper the authors proposes an update of the moving-average(?) every 20th iteration because it would have the same calculation overhead as RMSPROP. And still (I guess) better performance. I haven't found any good source of the implementation of the paper yet.  

- **ADAM**: – ADAptiv Momentum estimation – RMSprop + Stochastic Gradient Descent with momentum. ([youtube eve])  
<br/> theta_t+1 = theta_t - lr * momentum / denominator  
Uses 1st and 2nd momentum estimates. Adam also takes small steps in steep terrain and large steps in flat terrain. This is the result of using the denominator v_t^-(0.5).
average recent gradient: mom_t = beta1 * mom_t-1 + (1-beta1) * grad_t
average recent deviation in the gradient: v_t   = beta2 * v_t-1 + (1-beta2) * (grad_t)^2 
v_t is related to the second derivative and is in general close to constant. 
momentum    =  m_t 
denominator = (v_t)^-(0.5) + eps 

- **EVE**: – evolution of Adam () – locally and globaly adaptive learning-rate ([paper eve], [youtube eve])  
<br/> $$\theta_{t+1} = \theta_{t} - \frac{lr}{d_{t}} \frac{m_{t}}{denominator}$$
d_t is the only difference between Adam and Eve, has two objectives:  
**(i)** large variation in the Loss-function between steps should be given less weight -> take smaller steps.  
**(ii)** are we far from the minium (L*)? -> take larger steps.  
<br/> $\frac{1}{d_{t}} \propto \frac{L_{t} - L^{*}}{| L_{t} - L_{t-1} |}$  
problem (**ii**) If we step away from L* we might take incrementally larger and larger steps away from $L^{*}$ – blowing up.  
solution (**ii**) Clip the new term between $c$ and $\frac{1}{c}$.  
Also add smoothness to d_t with another running average (beta3).  
How to calculate the global minimum? Do Adam first and estimate the global minimum or set it to 0. 0 because it is the lower bound of the Loss-function.  


- SGRD:

- CLR – Cyclical Learning Rate:
Requires almost no additional computation 
Linear change of learning rate easiest to implement (my thoughts), chosen because any nonlinear change between largest and smallest gave the same result.

Iteration in one cycle (from paper) 4000, half cycle 2000 = stepsize.

    \# of iterations = trainingset-size / batch-size
    stepsize        = 2-10 times \# iterations in an epoch

    PyTorch:
      local cycle = math.floor(1 + epochCounter /(2∗ stepsize ))
      local x = math.abs(epochCounter/stepsize − 2∗cycle + 1)
      local lr = opt .LR + (maxLR − opt .LR) ∗ math.max(0, (1−x))

      variables:
        opt.LR is the specified lower (i.e., base) learning rate
        epochCounter is the number of epochs of training,
        lr is the computed learning rate

    Alternative methods:
      Linear2:   maximum learning rate is halfed after each cycle.
      exp_range: decrease both minimum and maximum boundary
                 by factor gamma^itteration after each cycle.





http://cs231n.github.io/neural-networks-3/                       <= gradient check
https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html  <= about floating point numbers


QUESTIONS:
	(i) If the regularization term depends on all weights in the network. Should it not be scaled with
		respect to the number of layers? If it is not, wouldn't the error blow up when starting to train,
		before the weights have possibility to change.

		What if we use a nonlinear function that regulates the lambda (regularization scaler)
		with respect to some global variable e.g batches trained or similar.

Adam Optimization:


Weight initialization:

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

  Xavier initialization – for tanh activation function
                          mean = zero
                          std of weights in layer i = 1/n_in – one over the
                          squareroot(number of inputs into that layer).

  He initialization     – for ReLU activation function
                          mean = zero
                          std of weights in layer i = sqrt (2/n_in)


Explaining Forward + Backwards pass:
	In the forward pass, regularization terms are not present.
	The cost/loss function introduces the regularization term. Before I thought that
	the regularization terms were only present when updating the weights but this is
	not true and it is logically incoherent. How could something show up in the changes
	of the weights if it was not present in the loss function? Could it?

	A loss function without regularization term would only depend on the changes of the
	weight depending on the activation... (I guess)


Vocabulary:
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

Activation functions:
	Sigmoid:
		If all inputs are positive, the derivative of the cost-function J for the 
		different layers are either all positive or all negative weights layerwise.
		I DON'T UNDERSTAND WHY. What is the derivative of dJ/dh? Where h is the
		sigmoid function? <- see slide 4 Josephine Sullivan.  
		
		cons:
			exp() is expensive to compute.
			outputs are not zero-centered.
			saturated activation kill the gradient. 
Inverse problem:
  input:  set of observations
  output: causal factors that produced them.
  calculate the causes with the result.

Regularization:
  Process of adding information to solve  an ill-posed problem or to prevent overfitting.
	Regularization can come about in many different ways. 
	Parameters that yield regularizing effects:
		* Large learning-rate 
		* Small batch sizes 
	When talking about regularization one typically means WeightDecay. 
  Weight decay exists as L1 and L2 regularization or both at the same time. 

	L1 – Lasso Regression – lambda sum 0i 
		Computational inefficient on non-sparse cases
	 	Sparse outputs
		Built in feature-selection	
		
		Shrinks less important feature's coefficients to zero. 
		
	L2 – Ridge Regression – lambda sum 0^2i
		Computational efficient due to having analytical solutions 
	    Non-sparse outputs 
		No feature-selection 
	
  loss function = sum {V(f(xi), yi)}

  cost function = loss function + regularization term = sum {V(f(xi), yi)} + lambda R(f)

  lambda controls the importance of the regularization term.

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
	
EMBEDDINGS
	combine categorical and continuous variables in a network:
		https://datascience.stackexchange.com/questions/29634/how-to-combine-categorical-and-continuous-input-features-for-neural-network-trai

Backpropagation:
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

RNNs:

	GRU – Gated Recurrent Unit
		A LSTM without an output-gate. And with a forget gate... (is that the same thing?) [Wikipedia]

  LSTM – Long Short Term Memory
    Stronger than GRUs, can easily perform unbounded counting (don't know what that entails) [Wikipedia]

		Both GRU and LSTM(not sure about this yet) can learn patterns that RNNs cannot learn.
		But cannot be trained parallelisation. No long and short range dependencies.

	Attention – (Fuzzy) Memory; Stanford course on NLP; Look it up.
		With attention units we can parallelise training.

	Self Attention –



  Transformers – avoids recurrence, uses attention. Allows parallelisation and faster training.
    relating signals from input and output positions depend only on "distance" between them.
    this linear dependence has a drawback – averaging attention-weighted positions – counteracted
    using Multi-Head Attention.

		Encoder:
			R    = [r1, r2, r3, r4] = d x 4					 <-- Word Embedding
			ai   =  R * softmax( (R' * ri)/sqrt(d) ) <-- Attention
		  dx1    dx4              4x1
			ri'  = max(W * ai + b, 0)   						 <-- (feedforward I think)

		Decoder:
			vi'  = max(W * bi + c, 0)

		Self attention between Encoder and Decoder:
			si   = R' softmax(R' * vi')

  	Multi-Head Attention –
			each head will make you focus on different things. (Like a convolution filter that focuses on different details?)
			for k = 1, ..., k
				- project your vi's with Wk -> vi^k  =   Wk 	 * 	 vi
				  														d0 x d	 d0 x d 		d x 1      d0 = d/k
				- then do self attention on     Vk   = [v1^k, v2^k, ..., v4^k]
				ai^k = Vk softmax(Vk^-1 vi^k) i=1, ..., 4
				vi^k' = ffnn(ai^k)

		Transformers also use Positional Encoding which is a sinusodial value
		depending on the position of the word in the senctence.

Papers to read:
	Improving Language Understanding by Generative Pre-Training
	BERT: Tre-training of Deep Bidirectional Transformers for Language Understanding
	Adversarial Examples Are Not Easily Detected: Bypassing Ten 




Standard machine translation metrics to evaluate results in NLP.



TRANSFORMERS:

  https://staff.fnwi.uva.nl/s.abnar/?p=108

  What is attention?
    http://jalammar.github.io/illustrated-transformer/



  Consists of:
    (i)  Encoders
    (ii) Decoders

WORD VECTORS:
  https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/

  Skip-Gram:
    Reverse of CBOW

  CBOW:
    Reverse of skip-gram


      Super Convergence:

        1cycle = curriculum learning and simulated annealing

        super convergence is "1cycle". Use only one cycle of CLR – clarification needed, does training stop at the top or bottom?


        In[1] it's stated that SC is possible if the LR-range test gives the result of a platoueing curve for larger learning-rate values. – clarify this even more!!!

        Better accuracy and faster learning compared to a solid learning rate over 10 times as many epochs.

        Best improvements when the ammount of training-data is limited.


        Super Convergence makes generalization better and the model requires less droput, smaller regularization terms,
        smaller weight decay.

        Weight decay of 10^-5 - 3*10^-6 is a guideline. Weight decay can then be adjusted for the model to not under- or over-fit.

LR range test:
  run model for several epochs over increasing learning rate values.
  max_lr  = lr(maximum accuracy) before accuracy get jagged/rough
  base_lr = lr(minimum accuracy) or 1/3 or 1/4 of max_lr


Loss Function Topology:
  Difficulty in minimizing the loss arrises
  from Saddle Points not  Poor Local Minima [Daupin et al.]

  Saddle points have small gradients that slow the learning process.
