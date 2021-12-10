X=3.14*rand(100,1);
Y=sin(2*X);
submodel_count=5;div_dimensions=[1];
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


X=(0:0.05:3.14)';
Y=sin(2*X);
result=zeros(1,size(X,1));
for i=1:1:size(X,1)
    result(i)=loli_out(sub_model,center,sigma,[1 X(i,:)],div_dimensions);
end
figure(2);
hold on
title('test set');
plot(X,result);
plot(X,Y);
legend('model','real');
hold off