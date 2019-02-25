%% Garch_LL_func.m
function G_LL_func = Garch_LL_func_11(theta,data)
    % the theta is (3)*1 
    % the data is n * 1
    % prepare the data for estimation of the LL function
    len = length(data);
    sigma2 = ones(len,1) * -1;
    % use the unconditional variance and real data to replace the conditional variance and y value of the first np terms 
    sigma2(1) = var(data);
    % get the conditional variance equation
    for i = 2:len
        sigma2(i) = theta(1) + theta(2) * data(i-1)^2 + theta(3) * sigma2(i-1); 
    end
    % get the LL function
    LL_func = sum((-1/2) * log(2 * pi) -1 * log(sqrt(sigma2))) - sum((data)).^2 ./ (2 * sigma2);
    G_LL_func = -1* LL_func / len;
end