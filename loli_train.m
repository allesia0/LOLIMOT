function [sub_model,center,sigma,left_range,right_range] = loli_train(Y_in,X_in,submodel_count,div_dimensions)
%LOLI_TRAIN 
% This function is used to train LOLIMOT model. Affine model(i.e.,y=ax+b) is used as local model.
% N=count of data in dataset. M=dimension of input.
% <Y_in>: train data y. Nx1 vector.
% <X_in>: train data x. NxM matrix.
% <submodel_count>: count of submodels
% <div_dimensions>: sometimes not all dimensions of input space are needed.
% This parameter controls which dimensions can be divided.
% y=x*w
%   Kernel(Weight) function is 
% $$e^{(x-c)\Sigma(x-c)^T }$$,or can be written as $$exp( \sum[\frac{x_i-c_i}{\sigma_i}]^2)$$
% only dimensions in <div_dimensions> would be divided
%%%%%%%%%%%
% some assist variables
input=X_in;
target=Y_in;
num_of_data=size(input,1);
num_of_input=size(input,2);
num_of_div_dimensions=length(div_dimensions);
div_input=input(:,div_dimensions);
affine_input=[ones(num_of_data,1) input];%used in affine model
local_err=zeros(1,submodel_count);
center=cell(1,submodel_count);
sigma=cell(1,submodel_count);
sub_model=cell(1,submodel_count);
left_range=cell(1,submodel_count);
right_range=cell(1,submodel_count);
% init first model
left_range{1}=min(div_input);
right_range{1}=max(div_input);
[sub_model{1},center{1},sigma{1},local_err(1)]=fitOneModel(target,affine_input,div_input,left_range{1},right_range{1});
% divide submodel_count-1 times
for i=2:1:submodel_count
    
    % find worst model to divide.
    [~,worst_llm_index]=max(local_err);
    %clear worst sub_model
    %In function find_best_divide, global error would be
    %calculated. In that function, worst submodel has been replaced by two
    %smaller model,so we need to clear worst sub_model first
    % function find_best_divide and loli_out will ignore empty submodel
    sub_model{worst_llm_index}=[];
    % find worst model's best divide.
    [best_divide_index]=find_best_divide(target,affine_input,div_input,left_range{worst_llm_index},right_range{worst_llm_index},div_dimensions,sub_model,center,sigma);
    %divide worst model according to best_divide
    [sub_model_temp,center_temp,sigma_temp,local_err_temp,left_range_temp,right_range_temp]=divide_submodel(target,affine_input,div_input,left_range{worst_llm_index},right_range{worst_llm_index},best_divide_index);
    % replace old model with two divide model
    sub_model{worst_llm_index}=sub_model_temp{1};
    center{worst_llm_index}=center_temp{1};
    sigma{worst_llm_index}=sigma_temp{1};
    local_err(worst_llm_index)=local_err_temp(1);
    left_range{worst_llm_index}=left_range_temp{1};
    right_range{worst_llm_index}=right_range_temp{1};
    sub_model{i}=sub_model_temp{2};
    center{i}=center_temp{2};
    sigma{i}=sigma_temp{2};
    local_err(i)=local_err_temp(2);
    left_range{i}=left_range_temp{2};
    right_range{i}=right_range_temp{2}; 
end
end
function[best_divide_index]=find_best_divide(target,affine_input,div_input,left_range,right_range,div_dimensions,sub_model_current,center_current,sigma_current)
% divide model along every dimension which allowed to divide.
% the best divide is the division which minimum the global error
    num_of_div_dimension=size(div_input,2);
    num_of_data=size(div_input,1);
    sub_model=cell(2,num_of_div_dimension);
    center=cell(2,num_of_div_dimension);
    sigma=cell(2,num_of_div_dimension);
    local_err=zeros(2,num_of_div_dimension);
    left_range_=cell(2,num_of_div_dimension);
    right_range_=cell(2,num_of_div_dimension);
    % divide along all dimension and get all divide results.
    for i=1:1:num_of_div_dimension
        [sub_model(:,i),center(:,i),sigma(:,i),local_err(:,i),left_range_(:,i),right_range_(:,i)]=divide_submodel(target,affine_input,div_input,left_range,right_range,i);
    end
    global_err=zeros(1,num_of_div_dimension);
    % calculate every division's global error.
    for i=1:1:num_of_div_dimension
        sub_model_temp=[sub_model_current sub_model(1,i) sub_model(2,i)];
        center_temp=[center_current center(1,i) center(2,i)];
        sigma_temp=[sigma_current sigma(1,i) sigma(2,i)];
        for j=1:1:num_of_data
            global_err(i)=global_err(i)+(target(j,:)-loli_out(sub_model_temp,center_temp,sigma_temp,affine_input(j,:),div_dimensions))^2;
        end
    end
    % find best divide index
    [~,best_divide_index]=min(global_err);
end



function[sub_model,center,sigma,local_err,left_range,right_range]=divide_submodel(target,affine_input,div_input,left_range_in,right_range_in,div_dimension_index)
left_range=cell(2,1);
right_range=cell(2,1);
sub_model=cell(2,1);
center=cell(2,1);
sigma=cell(2,1);
local_err=zeros(2,1);
div_center=(left_range_in(div_dimension_index)+right_range_in(div_dimension_index))/2;;
left_range{1}=left_range_in;
right_range{1}=right_range_in;
right_range{1}(div_dimension_index)=div_center;
left_range{2}=left_range_in;
left_range{2}(div_dimension_index)=div_center;
right_range{2}=right_range_in;
[sub_model{1},center{1},sigma{1},local_err(1)]=fitOneModel(target,affine_input,div_input,left_range{1},right_range{1});
[sub_model{2},center{2},sigma{2},local_err(2)]=fitOneModel(target,affine_input,div_input,left_range{2},right_range{2});
end
function [sub_model,center,sigma,local_err]=fitOneModel(target,affine_input,div_input,left_range,right_range)
num_of_data = size(target,1);  
center=(left_range+right_range)./2;
sigma=(right_range-left_range)./3;
weight=zeros(1,num_of_data);
for j=1:1:num_of_data
    temp=(div_input(j,:)-center)./sigma;
    weight(j)=exp(-0.5*(temp*temp'));
end
weight=diag(weight);
sub_model=inv(affine_input'*weight*affine_input)*affine_input'*weight*target;
err_temp=target-affine_input*sub_model;
local_err=err_temp'*weight*err_temp;
end

