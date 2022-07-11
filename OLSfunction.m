function params = OLSfunction(x,y)

% Add column for intercept
X = [ones(length(x),1), x]; 

% OLS
params = pinv(X'*X)*(X'*y);

end