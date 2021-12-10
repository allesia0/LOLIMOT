function [result]=loli_out(sub_model,center,sigma,affine_x_in,div_dimensions)
weight=zeros(1,length(sub_model));
origin_x_in=affine_x_in(:,2:end);
for i=1:1:length(sub_model)
    if isempty(sub_model{i})==false
        temp=(origin_x_in(1,div_dimensions)-center{i})./sigma{i};
        weight(i)=exp(-0.5*(temp*temp'));
    end
end
weight_total=sum(weight);
result=0;
for i=1:1:length(sub_model)
    if isempty(sub_model{i})==false
        result=result+weight(i)*affine_x_in*sub_model{i};
    end
end
result=result/weight_total;
end