# Local Linear Model Tree (LOLIMOT)
$$y=\sum_{i=1}^P e^{(x-c_i)\Sigma_i (x-c_i)^T}(xA+b)$$
where
$$\begin{aligned}
x&=1*M\ vector\\
y&=1*1\ vector\\
\Sigma_i&=diag([\frac{1}{\sigma_1^2},\frac{1}{\sigma_2^2},...,\frac{1}{\sigma_M^2}])
\end{aligned}$$

## Functions

#### [sub_model,center,sigma,left_range,right_range] = loli_train(Y_in,X_in,submodel_count,div_dimensions)

Train LOLIMOT model.

#### Input Parameters:

**Y_in**: N\*1 vector,dataset's output

**X_in**: N\*M matrix, dataset's input

**submodel_count**: count of submodels

**div_dimensions**: 1\*d vector. Sometimes not all dimension in the input space are needed to divide.

#### Output Parameters:

**sub_model**: 1\*P cell. 

**center**: 1\*P cell. The center of each submodel

**sigma**: 1\*P cell. Sigma of each submodel

**left_range**: 1\*P cell. left range of each submodel

**right_range**: 1\*P cell. right range of each submodel

Some output parameters are inputs of function loli_out


#### [result]=loli_out(sub_model,center,sigma,affine_x_in,div_dimensions)

LOLIMOT model out. 

#### Input pParameters:

**affine_x_in=[1 x_in], an 1\*(M+1) vector**

**sub_model**: Trained LOLIMOT submodel

**center**: output parameter of loli_train

**sigma**: output parameter of loli_train

**div_dimensions**: equals to loli_train's div_dimensions

#### Output pParameters:

**result**: model's output
