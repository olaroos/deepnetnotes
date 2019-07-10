[optimizing jupyter notebook]: <https://towardsdatascience.com/speed-up-jupyter-notebooks-20716cbe2025>
[keyword @property]: <https://www.programiz.com/python-programming/property>

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

##### [keyword @property]  


##### input parameters to functions:

def f(*args, \*\*kwargs):  

- args: all inputs that are not keywords i.e f(1,3,5)  
args = (1,3,5) 

- kwargs: all input that are given as keywords i.e f(input1=1,input2=3,input3=5) 
kwargs = {'input1':1,'input2':3,'input3':5} 

##### python magic commands:  

- __call__ called whenever a function is given a string as input. e.g f("hi")
