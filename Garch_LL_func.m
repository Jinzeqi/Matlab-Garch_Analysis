%Garch_LL_func.m
function G_LL_func = Garch_LL_func(theta,nq,data)
    % the theta is (1+q+p)*1 
    % the nq is the numebr of alpha
    % the data is n * 1
    % prepare the data for estimation of the LL function
    len = length(data);
    np = length(theta) - 1 - nq;
    max_pq = max(np,nq);
    sigma2 = ones(len,1) * -1;
    % use the unconditional variance and real data to replace the conditional variance and y value of the first np terms 
    sigma2(1:max_pq) = var(data);
    % get the conditional variance equation    
    for i = (max_pq+1):len
        sigma2(i) = theta(1:nq+1)' * [1;flip(data(i-nq:i-1))].^2 + theta(nq+2:end)' * flip(sigma2(i-np:i-1)); 
    end
    % get the LL function
    G_LL_func =  -1 * (sum((-1/2) * log(2 * pi) -1 * log(sqrt(sigma2))) - sum(data.^2 ./ (2 * sigma2))) / len;
end