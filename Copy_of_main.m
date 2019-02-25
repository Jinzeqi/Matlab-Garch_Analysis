clear;
clc;
%% 1. Construct the sample data path
number_value = 5000;
number_path = 2000;
true_theta = {0.05,0.06,0.3,0.5}';
nq = 2;
np = 1;
Garch21 = garch('Constant',cell2mat(true_theta(1)),'ARCH',true_theta(2:nq+1),'GARCH',true_theta(nq+2:end));
[~,simu_data_y] = simulate(Garch21,number_value,'Numpaths',number_path);
%% 2. estimate the theta
count_rej_ci = zeros(nq+np+1,1);
count_rej_p = zeros(nq+np+1,1);
true_theta = cell2mat(true_theta);
for i = 1:number_path
    init_theta = ones(nq+np+1,1) / (nq+np+1);
    [x,fval,~,~,~,hessian] = fminunc(@(x)Garch_LL_func(x,nq,simu_data_y(:,i)),init_theta);
    est_l = -1 * fval;
    % get the real part of the matrix which might have complex value
    sigma_theta = real((inv(hessian)/number_value)^0.5);
    len = length(x);
    %% 3.calculate 95% confidence interval and p value
    cl = 0.95;
    % use matrix operation to get the vector of l
    l = norminv(repelem(1-(1-cl)/2,len)',repelem(0,len)',diag(sigma_theta));
    % get the matrix of confidence interval; each row is one interval for theta
    ci = [x - l, x + l];
    % get the vector of p_values
    p_value = 2 * (1 - normcdf(abs(x - true_theta),repelem(0,len)',diag(sigma_theta)));
    count_rej_ci = count_rej_ci + ((true_theta < ci(:,1)) | (true_theta > ci(:,2)));
    count_rej_p = count_rej_p + (p_value < ones(np+nq+1,1) * (1-cl));
end
prob_ci = count_rej_ci ./ (ones(nq+np+1,1)*number_path)
prob_p = count_rej_p ./ (ones(nq+np+1,1)*number_path)