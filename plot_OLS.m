function fig = plot_OLS(x,y_observed,params)

% Get estimated params
alpha_hat = params(1);
beta_hat  = params(2);

% Get estimated linear fit
y_hat = alpha_hat + beta_hat*x;

% Plot the data
close;
fig = figure('color','w');
s1 = scatter(x,y_observed,30,'filled'); hold on;
p2 = plot(x,y_hat,'color','r','Linewidth',1.5);
leg_labels = {'Observed (noisy) data', ['OLS: \alpha = ' num2str(alpha_hat) ', \beta = ' num2str(beta_hat)] };
legend([s1,p2],leg_labels,'Box','off','Location','northwest')

end