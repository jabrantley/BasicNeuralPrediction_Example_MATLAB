% Example of neural prediction using EMG and movement data

% Load training data
load(fullfile(pwd,"data","train-emg.mat"));
load(fullfile(pwd,"data","train-kin.mat"));

% Define some colors
skyblue = [86,  180, 233]./256;
blue    = [0,   114, 178]./256;
green   = [0,   158, 115]./256;
pink    = [204, 121, 167]./256;

% Let's see what the data look like
fig = figure('color','white');
% EMG - fifth channel is channel we care about
subplot(2,1,1);
plot(EMG,'color','k');
subtitle('Right Knee EMG','fontweight','b')
xlim([0, size(EMG,2)]);
subplot(2,1,2);
plot(GONIO,'color','k');
subtitle('Right Knee Angle (goniometer measurement)','fontweight','b')
xlim([0, length(GONIO)]);

%% Out of curiosity, let's see what the relationship between the raw EMG and movement data looks like
close;
fig = figure('color','white');
scatter(EMG,GONIO,10,'k');
xlabel('Right Knee EMG')
ylabel('Right Knee Angle')
box on;
title({['Clearly from this figure we can see that the raw EMG'],['has very little predictive power for predicting movement']})
       
%% Lets start with some basic data processing
close;

% The movements are happening at approximately 1 Hz, so lets apply a low pass filter to the EMG
% To get us close to this frequency to get the envelope. First take
% absolute value then apply low pass filter
low_pass_freq = 10;
srate = 1000;
[b,a] = butter(2,low_pass_freq/(srate/2),'low');
ENV = filtfilt(b,a, abs(EMG));

% Let's see what the data look like
fig = figure('color','white');
% EMG - fifth channel is channel we care about
subplot(2,1,1);
plot(EMG,'color',0.5.*ones(3,1)); hold on;
plot(abs(EMG),'color',skyblue); hold on;
plot(ENV,'color',blue,'linewidth',2)
leg = legend({"Raw EMG", "Abs EMG", "EMG Envelope"},'Location','northoutside','Orientation','horizontal','box','off');
title(leg,'Right Knee EMG','fontweight','b','fontsize',10)
xlim([0, size(EMG,2)]);
subplot(2,1,2);
plot(GONIO,'color','k');
subtitle('Right Knee Angle (goniometer measurement)','fontweight','b','fontsize',10)
xlim([0, length(GONIO)]);


%% We can play with the low pass to see how it impacts the envelope
close;

% Let's see what the data look like
fig = figure('color','white');

% First low pass filter
subplot(4,1,1);

low_pass_freq = 10;
srate = 1000;
[b,a] = butter(2,low_pass_freq/(srate/2),'low');
ENV = filtfilt(b,a, abs(EMG));

plot(EMG,'color',0.7.*ones(3,1)); hold on;
plot(ENV,'color',pink,'linewidth',3)
subtitle(['Right Knee EMG (', num2str(low_pass_freq) ' Hz low pass)'],'fontweight','b','fontsize',10)
xlim([0, size(EMG,2)]);

% Second low pass filter
subplot(4,1,2);

low_pass_freq = 5;
srate = 1000;
[b,a] = butter(2,low_pass_freq/(srate/2),'low');
ENV = filtfilt(b,a, abs(EMG));

plot(EMG,'color',0.7.*ones(3,1)); hold on;
plot(ENV,'color',blue,'linewidth',3)
subtitle(['Right Knee EMG (', num2str(low_pass_freq) ' Hz low pass)'],'fontweight','b','fontsize',10)
xlim([0, size(EMG,2)]);

% Second low pass filter
subplot(4,1,3);

low_pass_freq = 2;
srate = 1000;
[b,a] = butter(2,low_pass_freq/(srate/2),'low');
ENV = filtfilt(b,a, abs(EMG));

plot(EMG,'color',0.7.*ones(3,1)); hold on;
plot(ENV,'color',green,'linewidth',3)
subtitle(['Right Knee EMG (', num2str(low_pass_freq) ' Hz low pass)'],'fontweight','b','fontsize',10)
xlim([0, size(EMG,2)]);

subplot(4,1,4);
plot(GONIO,'color','k');
subtitle('Right Knee Angle (goniometer measurement)','fontweight','b','fontsize',10)
xlim([0, length(GONIO)]);



%% Now let's see the relationship between the envelope and movement data

close;

low_pass_freq = 5;
srate = 1000;
[b,a] = butter(2,low_pass_freq/(srate/2),'low');
ENV = filtfilt(b,a, abs(EMG));

fig = figure('color','white');
scatter(ENV,GONIO,10,'k');
xlabel('Right Knee EMG')
ylabel('Right Knee Angle')
box on;
title({['We a slight linear relationship, but there is clearly some hysteresis.'],['This implies some kind of nonlinearity in the data.']})

%% Nonetheless, let's go ahead and try to use a linear model to predict the data
close; 

% We will use our ols_class to do this
obj = OLSclass(ENV,GONIO);
obj.train()

% Plot the data
fig = obj.plot_OLS();
subtitle(['Pearson correlation coefficient: ', num2str(obj.PearsonCorr(GONIO,obj.y_hat))],'fontweight','bold');


%% Now what does this look like for prediction

close;
fig = figure('color','white');

plot(GONIO,'color',0.5.*ones(3,1)); hold on;
plot(obj.y_hat,'color',pink','linewidth',2)
legend({'True joint angle', 'Predicted Joint Angle'},'box','off')

SSE = sum((obj.y-obj.y_hat).^2);   % Sum of squared error
SST = sum((obj.y-mean(obj.y)).^2); % Sum of squared total
R2  = 1-SSE/SST;
                
subtitle(['R^2: ', num2str(R2)],'fontweight','bold');




%% Now lets try the whole process on test data
temp = struct2cell(load(fullfile(pwd,"data","test-emg.mat")));
test.EMG = temp{1};
temp = struct2cell(load(fullfile(pwd,"data","test-kin.mat")));
test.GONIO = temp{1}; clear temp;

% Apply filter
test.ENV = filtfilt(b,a, abs(test.EMG));

close;
fig = figure('color','white');

% Test EMG data
subplot(2,1,1);
plot(test.EMG,'color',0.5.*ones(3,1)); hold on;
plot(abs(test.EMG),'color',skyblue); hold on;
plot(test.ENV,'color',blue,'linewidth',2)
leg = legend({"Raw EMG", "Abs EMG", "EMG Envelope"},'Location','northoutside','Orientation','horizontal','box','off');
title(leg,'Right Knee EMG Test Data','fontweight','b','fontsize',10)
xlim([0, size(test.EMG,2)]);

subplot(2,1,2);
plot(test.GONIO,'color',0.5.*ones(3,1)); hold on;
gonio_predict = obj.predict(test.ENV);
plot(gonio_predict,'color',pink','linewidth',2)
xlim([0, length(test.GONIO)]);
leg = legend({'True joint angle', 'Predicted Joint Angle'},'Location','northoutside','Orientation','horizontal','box','off');

SSE = sum((test.GONIO(:)-gonio_predict).^2);   % Sum of squared error
SST = sum((test.GONIO(:)-mean(test.GONIO(:))).^2); % Sum of squared total
R2  = 1-SSE/SST;
title(leg,['Raw vs Predicted Joint Angle on Test Data (R^2: ', num2str(R2) ')'],'fontweight','bold','fontsize',10);
% subtitle('Right Knee Angle (goniometer measurement)','fontweight','b','fontsize',10)


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                            %
%               NOW LETS TRY A KALMAN FILTER                 %
%                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Kalman filter parameters
KF_ORDER  = 1;
KF_LAGS   = 1;
KF_LAMBDA = 0.1'; %logspace(-2,2,5);

% Reduce sampling rate to speed up training - our frequency of interest is 
% so low that this does not hurt at all. 
GONIO_DS = decimate(GONIO(:),20);
ENV_DS   = decimate(ENV(:),20);

KF = KalmanFilter('state',GONIO_DS(:)','observation',ENV_DS','augmented',0,...
    'method','normal','lags',KF_LAGS,'order',KF_ORDER,'lambdaF',KF_LAMBDA,'lambdaB',KF_LAMBDA);

KF.train()

gonio_predict = zeros(length(ENV_DS),1);
for ii = 1:length(gonio_predict)
    gonio_predict(ii) = KF.predict(ENV_DS(ii)); 
end


close;
fig = figure('color','white');

% Test EMG data
subplot(2,1,1);
plot(decimate(EMG,20),'color',0.5.*ones(3,1)); hold on;
plot(abs(decimate(EMG,20)),'color',skyblue); hold on;
plot(ENV_DS,'color',blue,'linewidth',2)
leg = legend({"Raw EMG", "Abs EMG", "EMG Envelope"},'Location','northoutside','Orientation','horizontal','box','off');
title(leg,'Right Knee EMG Training Data','fontweight','b','fontsize',10)
xlim([0, length(ENV_DS)]);

subplot(2,1,2);
plot(GONIO_DS,'color',0.5.*ones(3,1)); hold on;
plot(gonio_predict,'color',pink','linewidth',2)
xlim([0, length(GONIO_DS)]);
leg = legend({'True joint angle', 'Predicted Joint Angle'},'Location','northoutside','Orientation','horizontal','box','off');

SSE = sum((GONIO_DS - gonio_predict).^2);   % Sum of squared error
SST = sum((GONIO_DS(:)-mean(GONIO_DS(:))).^2); % Sum of squared total
R2  = 1-SSE/SST;
title(leg,['Raw vs Predicted Joint Angle on Training Data (R^2: ', num2str(R2) ')'],'fontweight','bold','fontsize',10);


%% Now lets do the same thing on the test data
test.GONIO_DS = decimate(test.GONIO,20);
test.ENV_DS = decimate(test.ENV,20);

gonio_predict = zeros(length(test.ENV_DS),1);
for ii = 1:length(gonio_predict)
    gonio_predict(ii) = KF.predict(test.ENV_DS(ii)); 
end


close;
fig = figure('color','white');

% Test EMG data
subplot(2,1,1);
plot(decimate(test.EMG,20),'color',0.5.*ones(3,1)); hold on;
plot(abs(decimate(test.EMG,20)),'color',skyblue); hold on;
plot(test.ENV_DS,'color',blue,'linewidth',2)
leg = legend({"Raw EMG", "Abs EMG", "EMG Envelope"},'Location','northoutside','Orientation','horizontal','box','off');
title(leg,'Right Knee EMG Training Data','fontweight','b','fontsize',10)
xlim([0, length(test.ENV_DS)]);

subplot(2,1,2);
plot(test.GONIO_DS,'color',0.5.*ones(3,1)); hold on;
plot(gonio_predict,'color',pink','linewidth',2)
xlim([0, length(test.GONIO_DS)]);
leg = legend({'True joint angle', 'Predicted Joint Angle'},'Location','northoutside','Orientation','horizontal','box','off');

SSE = sum((test.GONIO_DS(:) - gonio_predict).^2);   % Sum of squared error
SST = sum((test.GONIO_DS(:)-mean(test.GONIO_DS(:))).^2); % Sum of squared total
R2  = 1-SSE/SST;
title(leg,['Raw vs Predicted Joint Angle on Training Data (R^2: ', num2str(R2) ')'],'fontweight','bold','fontsize',10);



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                            %
%          NOW LETS TRY A KALMAN FILTER GRID SEARCH          %
%                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close;

% Kalman filter parameters
KF_ORDER  = 1;
KF_LAGS   = 1;
KF_LAMBDA = logspace(-2,2,5);

% Reduce sampling rate to speed up training - our frequency of interest is 
% so low that this does not hurt at all. 
GONIO_DS = decimate(GONIO(:),20);
ENV_DS   = decimate(ENV(:),20);

KF = KalmanFilter('state',GONIO_DS(:)','observation',ENV_DS','augmented',0,...
    'method','normal');

% Perform grid search
KF.grid_search('order',KF_ORDER,'lags',KF_LAGS,'lambdaB',KF_LAMBDA,'lambdaF',KF_LAMBDA,'kfold',3,'testidx',1);


gonio_predict = zeros(length(test.ENV_DS),1);
for ii = 1:length(gonio_predict)
    gonio_predict(ii) = KF.predict(test.ENV_DS(ii)); 
end

fig = figure('color','white');

% Test EMG data
subplot(2,1,1);
plot(decimate(test.EMG,20),'color',0.5.*ones(3,1)); hold on;
plot(abs(decimate(test.EMG,20)),'color',skyblue); hold on;
plot(test.ENV_DS,'color',blue,'linewidth',2)
leg = legend({"Raw EMG", "Abs EMG", "EMG Envelope"},'Location','northoutside','Orientation','horizontal','box','off');
title(leg,'Right Knee EMG Training Data','fontweight','b','fontsize',10)
xlim([0, length(test.ENV_DS)]);

subplot(2,1,2);
plot(test.GONIO_DS,'color',0.5.*ones(3,1)); hold on;
plot(gonio_predict,'color',pink','linewidth',2)
xlim([0, length(test.GONIO_DS)]);
leg = legend({'True joint angle', 'Predicted Joint Angle'},'Location','northoutside','Orientation','horizontal','box','off');

SSE = sum((test.GONIO_DS(:) - gonio_predict).^2);   % Sum of squared error
SST = sum((test.GONIO_DS(:)-mean(test.GONIO_DS(:))).^2); % Sum of squared total
R2  = 1-SSE/SST;
title(leg,['Raw vs Predicted Joint Angle on Training Data (R^2: ', num2str(R2) ')'],'fontweight','bold','fontsize',10);


% Add fig title
fig.Children(4).Title.String = 'Results of KF with paramter grid search on test data.';
fig.Children(4).Title.Units = 'normalized';
fig.Children(4).Title.Position(2) = 1.75;
fig.Children(4).Title.FontSize = 12;

