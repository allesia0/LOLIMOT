X=3.14*rand(100,3);
Y=sin(2*X(:,1))+X(:,2).*X(:,3);
submodel_count=5;% 5 submodels
div_dimensions=[1,3];% only dimension 1 and 3 would be divided
[sub_model,center,sigma,left_range,right_range]=loli_train(Y,X,submodel_count,div_dimensions);
result=zeros(1,size(X,1));
for i=1:1:size(X,1)
    result(i)=loli_out(sub_model,center,sigma,[1 X(i,:)],div_dimensions);
end
figure(1);
hold on
title('training set');
plot(result);
plot(Y);
legend('model','real');
hold off


X=3.14*rand(100,3);
Y=sin(2*X(:,1))+X(:,2).*X(:,3);
result=zeros(1,size(X,1));
for i=1:1:size(X,1)
    result(i)=loli_out(sub_model,center,sigma,[1 X(i,:)],div_dimensions);
end
figure(2);
hold on
title('test set');
plot(result);
plot(Y);
legend('model','real');
hold off