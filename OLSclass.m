% ************************************************************************
% Author:     Justin A Brantley
% Date:       07/11/2022
% Comments:   This implements OLS
% ************************************************************************

classdef OLSclass < handle
    
    % These are things I want users to be able to see but not change
    properties (SetAccess = private, GetAccess = public)
        X
        params
        y_hat
    end
    
    % These are things I want people to see AND change
    properties (SetAccess = public, GetAccess = public)
        x
        y
    end
        
    % Now we have all of the methods, or functions, that can operate on our
    % class
    methods (Access = public)
        
        % We begin by building our class internally. This is called the
        % constructor..
        function self = OLSclass(x,y)
            % Data
            self.x = x(:);
            self.y = y(:);
            
            % Add column for intercept
            self.X = [ones(length(self.x),1), self.x]; 
        end
            
        % Perform OLS regression
        function train(self) % <-- we don't pass x,y because they are already properties of the object
            % OLS
            self.params = pinv(self.X'*self.X)*(self.X'*self.y);
            % We can use our own internal functions here
            self.y_hat = self.test();
        end % Notice there are not ouptuts since we are just assigning the result back to the obj
        
        % Test our data - just realized this is redundant with predict
        function y_test = test(self)
              y_test = self.predict(self.x);   
        end
        
        % If we have hold out data we can test it
        function y_predict = predict(self,x_predict)
              y_predict = self.params(1) + x_predict(:)*self.params(2);   
        end
        
        function fig = plot_OLS(self)
            
            % Plot the data
            fig = figure('color','w');
            s1 = scatter(self.x,self.y,30,0.5.*ones(length(self.x),3),'filled'); hold on;
            if ~isempty(self.y_hat)
                p1 = plot(self.x,self.y_hat,'color',[0,114, 178]./256','Linewidth',1.5);
                which_leg = [s1,p1];
                leg_labels = {'Observed (noisy) data', ['OLS: \alpha = ' num2str(self.params(1)) ', \beta = ' num2str(self.params(2))] };
            else
                which_leg = s1;
                leg_labels = 'Observed (noisy) data';
            end
            
            legend(which_leg,leg_labels,'Box','off','Location','best');
        end
    end % end public methods
           
    % Static methods - these do not rely on the object inself. You can
    % think of these are just regular functions
    methods (Static)
        function r = PearsonCorr(x,y)
            % Calculate the Pearson correlation coefficient between vectors x and y
            x = x(:);
            y = y(:);
            % Get size of data
            [n,~] = size(x);
            [m,~] = size(y);
            % Compute r value
            if n == m
                mu_x = mean(x);
                mu_y = mean(y);
                % Get num
                num = (x - mu_x)' * (y - mu_y);
                % Get denom
                denom = sqrt(sum((x - mu_x).^2)) * sqrt(sum((y - mu_y).^2));
                % Compute r-value
                r = num/denom;
            else
                error('X and Y must be the same length.')
            end
        end
    end % end static methods
end
  
            
           
        
        
        
        
        
