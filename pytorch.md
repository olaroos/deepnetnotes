[optimizing jupyter notebook] <https://towardsdatascience.com/speed-up-jupyter-notebooks-20716cbe2025>

#### 

##### nn.ModuleList  
layers = [Conv2 Linear Relu]  
nn.ModuleList(layers)  

ModuleList gives names to layers and returns a struct with them attatched.  

##### torch.no_grad()  
This function should be used when updating parameters through a custom function. This is because that operation shouldn't  
be part of the next gradient calculation.  
It should also be used when doing the evaluation – calculating the validation loss.  

##### python yield  
yield returns a Generator


##### [optimizing jupyter notebook]
