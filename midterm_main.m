clear
clc
%% 1. read the price data and transform to log return
price = xlsread('PriceSeries1.xls');
ret = price2ret(price);
len = length(ret);
%% 2. Pre-Estimation Process
% plot return data
figure(1)
plot(ret)
index = [1,round(len / 4),round(len * 2 / 4),round(len * 3 / 4),len];
set(gca,'XTick',index)
set(gca,'XTickLabel',index)
% Convert y-axis values to percentage values by multiplication
a=[cellstr(num2str(get(gca,'ytick')'*100))];
% Create a vector of '%' signs
pct = char(ones(size(a,1),1)*'%'); 
% Append the '%' signs after the percentage values
new_yticks = [char(a),pct];
set(gca,'yTickLabel',new_yticks)
ylabel('Daily Return')
title(' Ticker Daily Return')
% check for correlation in return and plot
figure(2)       
subplot(2,1,1)
autocorr(ret)
title('ACF for daily return series')
subplot(2,1,2)
parcorr(ret)
title('PACF for daily return series')
% check for correlation in the squared return and plot
figure(3)
subplot(2,1,1)
autocorr(ret.^2)
title('ACF for daily squared return series')
subplot(2,1,2)
parcorr(ret.^2)
title('PACF for daily squared return series')
% conduct Q TEST
[lbq_ret,p_ret,stat_ret,cri_ret] = lbqtest((ret-mean(ret)),'Lags',[5,10,15],'Alpha',0.05);
[lbq_ret2,p_ret2,stat_ret2,cri_ret2] = lbqtest((ret-mean(ret)).^2,'Lags',[5,10,15],'Alpha',0.05);
% conduct Arch Test
[h_arch_ret,p_arch_ret,stat_arch_ret,cri_arch_ret] = archtest((ret-mean(ret)),'Lags',[5,10,15],'Alpha',0.05);

%% 3. Parameter-Estimation Process
% use garch(1,1) model
garch11 = garch('GARCHLags',1,'ARCHLags',1);
[fit11,~,L11,~] = estimate(garch11,ret);
% use garch(2,1) model
garch21 = garch('GARCHLags',2,'ARCHLags',1);
[fit21,~,L21,~] = estimate(garch21,ret);
% use garch(1,2) model
garch12 = garch('GARCHLags',1,'ARCHLags',2);
[fit12,~,L12,~] = estimate(garch12,ret);
% find the best model with lowest aic value
inf_cri = aicbic([L11,L12,L21],[3,4,4],[len,len,len]);
%% 4. Post-Estimation Process
% get the conditional variance and residuals
[cv11,res11] = simulate(fit11,len);
% plot conditional variance data
figure(4)
subplot(3,1,1)
plot(cv11)
set(gca,'XTick',index)
set(gca,'XTickLabel',index)
ylabel('Conditional Variances')
title(' Ticker Daily Return Conditional Variances by Garch(1,1)')
% plot residual / innovation data
subplot(3,1,2)
plot(res11)
set(gca,'XTick',index)
set(gca,'XTickLabel',index)
ylabel('Residuals')
title(' Ticker Daily Return by Garch(1,1)')
% plot Return data
subplot(3,1,3)
plot(res11 + fit11.Constant)
set(gca,'XTick',index)
set(gca,'XTickLabel',index)
ylabel('Return')
title(' Ticker Daily Return by Garch(1,1)')
% check for correlation for standadized residuals 
figure(5)
subplot(2,1,1)
autocorr((res11./sqrt(cv11)).^2)
title('ACF for daily squared standadized residuals series')
subplot(2,1,2)
parcorr((res11./sqrt(cv11)).^2)
title('PACF for daily squared standadized residuals series')
% conduct Q TEST for standadized residuals 
[lbq_res,p_res,stat_res,cri_res] = lbqtest((res11./sqrt(cv11)).^2,'Lags',[5,10,15],'Alpha',0.05);
% conduct Arch Test for standadized residuals 
[h_arch_res,p_arch_res,stat_arch_res,cri_arch_res] = archtest((res11./sqrt(cv11)),'Lags',[5,10,15],'Alpha',0.05);
% forecast 1 period ahead conditional variance
fore_vol = forecast(fit11,1,'Y0',ret);