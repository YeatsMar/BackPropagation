# BackPropagration Network
## Principle
### Forward
![basic_BP](forREADME/basic_BP.png)  
Each unit (or say, neuron) of the input layer contributes  to its next neuron, i.e. each unit of next layer (output layer), to some extent. The extents (weights) of the contributions are independent. Thus the connection between input layer and output layer is fully connected and single directed.  

Take the basic demo BP network in the above picture as an example to illustrate:  
1. the input is composed of 2 units  $X = [x_{1},   x_{2}]$  
2. and the output has 2 units as well  $Y = [A, B]$   
3. then the weights should be a 2 $\times$ 2 matrix $\theta$ = [$w_{1A}$ $x_{1B}$; $w_{2A}$ $x_{2B}$]  
4. fully-connected contribution means $Y =$ $\theta^T$$X$, i.e. $A =$ $w_{1A}$$x_{1}$ + $w_{2A}$$x_{2}$, $B =$ $w_{1B}$$x_{1}$ + $w_{2B}$$x_{2}$.  
5. $Y$ need post-process to be actuall output: $Y = f(X)$. $f$ is an activation function, which can be *sigmoid, tanH, ReLu, softmax* and other **non-linear** functions.  

The last procedure is necessary to build a complicated network with the ability of self learning. If the function is linear or there is no such a function, the output is just a linear expression of input. Then the deviation of the error on weights are constant, leading to disability of self adjustment of weights to minimize its error. Refer to biological nerual network in human brain, the activation function is what it is called. A neuron is only actived when its shreshold is achieved.  
![bias](forREADME/bias.png)  
In practice, we actually add a bias unit in the input to better fit a AI model. For convenience, I treat bias as $x_{0}$ and set $x_{0}$ = 1, then let its weight $w_{0x}$ to modify the actual value as bias of the input layer.  

For a multi-layer network, the output $Y$ will be input of next layer. The number of layers can be any positive integer.  
![layers](forREADME/layers.png)  

### Backward  
By calculating the error and formatting it, the error of each unit can be derived. Then take advantage of *Gradient Descent*, the weights will be adjusted to achieve optimal network step by step, or technically say epoch by epoch.  

Since the error is calculated first on the last layer and then second layer and then gradually until the second layer (the first layer is just input, so calculating its error does not make sense), the process is called backward propagation of errors. That is why BP network is so named.  

![output1](forREADME/output1.png)  

Represent the process of an epoch in Math goes like this:  
1. Calculate the value at the last layer $\frac {\partial Error}{\partial O^{(L)} }$， $L$ is the number of layers and $O$ is the output of a single unit at a layer. The value and format depends on different problems and will be discussed in details later.  
2. Then goes to previous layer:$$\frac{\partial Error}{\partial O^{(L-1)}}=\frac{\partial Error}{\partial O^{(L)}}\frac{\partial O^{(L)}}{\partial O^{(L-1)}}$$ Because $$O^{(L)} = f(g), g = \sum w^{(L-1)} O^{(L-1)}$$ $$\frac{\partial O^{(L)}}{\partial O^{(L-1)}} = \frac{\partial f(\sum w^{(L-1)} O^{(L-1)})}{\partial O^{(L-1)}} = f^{\prime}(g)* \sum w^{(L-1)}$$Thus$$\frac{\partial Error}{\partial O^{(L-1)}}=\frac{\partial Error}{\partial O^{(L)}}*f^{\prime}(g)* \sum w^{(L-1)}$$$f^{\prime}(g)$ is the deviation of activation function and varies.   
3. Define $$\delta^{(l)}=\frac{\partial Error}{\partial O^{(l)}}*f^{\prime}(g)$$Then the chain rule can be presented as:$$\delta^{(l)}=\delta^{(l+1)}f^{\prime}(g)\sum w^{(l)}, l=2,3,...,L-1$$
4. Adjust weights:$$\Delta w_{ji}^{(l)}=\alpha*\frac{\partial Error}{\partial w_{ji}^{(l)}}$$$\alpha$ is the learning rate. And $$\frac{\partial Error}{\partial w_{ji}^{(l)}}=\frac{\partial Error}{\partial O_{i}^{(l+1)}}\frac{\partial O_{i}^{(l+1)}}{\partial w_{ji}^{(l)}}$$ $$\frac{\partial O_{i}^{(l+1)}}{\partial w_{ji}^{(l)}} = \frac{\partial f(\sum w_{i}^{(l)} O_{i}^{(l)})}{\partial w_{ji}^{(l)}} = f^{\prime}(g)*O_{i}^{(l)}$$ Thus, $$\Delta w_{ji}^{(l)}=\alpha*\frac{\partial Error}{\partial O_{i}^{(l+1)}}*f^{\prime}(g)*O_{i}^{(l)}=\alpha\delta^{(l)}O_{i}$$ In practice, we often adjust weights layer by layer once the needed $\delta$ is derived. And since $\delta$ is derived from last layer to second layer, the asjustment of weights starts from the last second layer to the first layer - **backward propagation**. And the number of epochs should be large enough to achieve little errors.

#### Regression
In regression problems, the error or cost of the final output is square mean: (m is the number of samples)
$$
Error = \frac{1}{2}\sum_{i=1}^{m}(d-O)^2
$$
$m$ is the number of samples, $d$ is the should-be value, $O$ is the predicted value, i.e. the output of last layer. 
 
Then its partial derivative with respect to the outputs: 
$$
\frac{\partial Error}{\partial O} = \frac{\partial \frac{1}{2}\sum_{i=1}^{m}(d-O)^2}{\partial O} = \frac{1}{2}\sum_{i=1}^{m}\frac{\partial (d-O)^2}{\partial O} = \sum_{i=1}^{m}(O - d)
$$  
And $$\delta^{(L)}=\sum_{i=1}^{m}(O - d)*f^{\prime}(g)$$  

#### Classification  
In classification problems, the error or cost of the final output is cross entropy:$$Error=\sum_{i=1}^{m}d*lnO$$ Both $d$ and $O$ can only be 0 or 1.  
If it is a logistic classfication, the partial derivative of error with respect to the outputs is also: $$\frac{\partial Error}{\partial O}=\sum_{i=1}^{m}(O - d)$$ The same as regression problems.  
Proof:  
$$Error=\sum_{i=1}^{m}d*lnO=\sum_{i=1}^{m}[d*lnO+(1-d)*ln(1-O)]$$
$$=\sum_{i=1}^{m}\{d*[lnO-ln(1-O)]+ln(1-O)\}$$
$$=\sum_{i=1}^{m}[d*ln\frac{O}{1-O}+ln(1-O)]$$
Because
$$O=sigmoid(\Theta^TX)=\frac{1}{1+e^{-\Theta^T X}}=\frac{e^{\Theta^T X}}{e^{\Theta^T X}+1}$$
$$1-O=\frac{1}{e^{\Theta^T X}+1}$$
$$ln\frac{O}{1-O}=ln(e^{\Theta^TX})=\Theta^TX$$
$$ln(1-O)=-ln(\frac{1}{1-O})=-ln(1+e^{\Theta^TX})$$
Thus
$$Error=\sum_{i=1}^{m}[d*\Theta^TX-ln(1+e^{\Theta^TX})]$$
$$\frac{\partial Error}{\partial \Theta}=\sum_{i=1}^{m}[d*X-\frac{e^{\Theta^TX}}{1+e^{\Theta^TX}}*X]=$$