clear;
clc;
%% 1. Construct the sample data path
number_value = 10000;
number_path = 10000;
true_theta = [0.05,0.2,0.6]';
Garch11 = garch('Constant',true_theta(1),'ARCH',true_theta(2),'GARCH',true_theta(3));
[~,simu_data_y] = simulate(Garch11,number_value,'Numpaths',number_path);
%% 2. estimate the theta
% this is the rejection number counted by confidence interval
count_rej_ci = [0;0;0];
% this is the rejection number counted by p-value
count_rej_p = [0;0;0];
for i = 1:number_path
    init_theta = [0.1;0.1;0.2];
    [x,fval,~,~,~,hessian] = fminunc(@(x)Garch_LL_func_11(x,simu_data_y(:,i)),init_theta);
    est_l = -1 * fval;
    % get the real part of the matrix which might have complex value
    % hessian already divided by number in the function of likelihood
    sigma_theta = real(sqrt(inv(hessian)/number_value));
    len = length(x);
    %% 3.calculate 95% confidence interval and p value
    cl = 0.95;
    % use matrix operation to get the vector of l
    l = norminv(repelem(1-(1-cl)/2,len)',repelem(0,len)',diag(sigma_theta));
    % get the matrix of confidence interval; each row is one interval for theta
    ci = [x - l, x + l];
    % get the vector of p_values
    p_value = 2 * (1 - normcdf(abs(x - true_theta),repelem(0,len)',diag(sigma_theta)));
    % coutn by 1 if the true_theta is not between ci
    count_rej_ci = count_rej_ci + ((true_theta < ci(:,1)) | (true_theta > ci(:,2)));
    % coutn by 1 if p-value is less than confidence level    
    count_rej_p = count_rej_p + (p_value < [1-cl;1-cl;1-cl]);
end
%calculate the probalility of rejection rate by confidence interval
prob_ci = count_rej_ci ./ [number_path;number_path;number_path]
%calculate the probalility of rejection rate by p-value
prob_p = count_rej_p ./ [number_path;number_path;number_path]