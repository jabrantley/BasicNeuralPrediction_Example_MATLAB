% Example of using OOP on a simple linear regression problem

% Data
x = [0:.005:1]';
alpha = 1;              % Intercept
beta  = 2;              % Slope
e = 0.25*randn(1,length(x)); % Noise

% Define equation
y_true     = alpha + beta*x;
y_observed = y_true + e(:);

% Plot the data
fig = figure('color','w');
s1 = scatter(x,y_observed,30,'filled'); hold on;
p1 = plot(x,y_true,'color','k','Linewidth',1.5);
legend([s1,p1],{'Observed (noisy) data',['True data: \alpha = ' num2str(alpha) ', \beta = ' num2str(beta)] },'Box','off','Location','northwest')

%% Perform OLS regression
X = [ones(length(x),1), x]; % Add column for intercept
params = pinv(X'*X)*(X'*y_observed);

% Get estimated params
alpha_hat = params(1);
beta_hat  = params(2);

% Get estimated linear fit
y_hat = alpha_hat + beta_hat*x;


% Plot the data
close;
fig = figure('color','w');
s1 = scatter(x,y_observed,30,'filled'); hold on;
p1 = plot(x,y_true,'color','k','Linewidth',1.5);
p2 = plot(x,y_hat,'color','r','Linewidth',1.5);
leg_labels = {'Observed (noisy) data',['True data: \alpha = ' num2str(alpha) ', \beta = ' num2str(beta)], ['OLS: \alpha = ' num2str(alpha_hat) ', \beta = ' num2str(beta_hat)] };
legend([s1,p1,p2],leg_labels,'Box','off','Location','northwest')


%% Now consider that we can just write the two things we do as functions

% Get params
params = OLSfunction(x,y_observed)

% Plot data - note that in our function we are pretending like we don't 
% know the true data
fig = plot_OLS(x,y_observed,params);

% Of course we can add it if we wanted...
hold on; 
p1 = plot(x,y_true,'color','k','Linewidth',1.5);


%% Lets try writing an OLS class

% The passing of params to various functions can get cumbrersome if you
% need to pass alot of things around. We can instead write an OLS class
% that can have properties and methods

% Define class
obj = OLSclass(x,y_observed)

% Notice (in the command window) that obj has x,y, and X but not params/
% y_hat since we have not trained the model yet.

%% Plot the data
fig = obj.plot_OLS();

%% Run regression 
close; 

% No need to pass the data because its been assigned above. We execute the 
% function but just calling it. 
obj.train()

% Now take a look at the object again and see that everything is there
obj

%% Correlation
% We can check the correlation coefficient between our actual data and 
% our noisy data/ true data.

% Correlation between noisy observed data and estimated data
r1 = obj.PearsonCorr(y_observed,obj.y_hat)

%% Plot the data
fig = obj.plot_OLS();

%% Add true data to fig object output from our class method
plot(x,y_true,'color','k','Linewidth',1.5,'DisplayName',['True data: \alpha = ' num2str(alpha) ', \beta = ' num2str(beta)])
