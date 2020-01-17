function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_minus_y = (theta' * X')' - y;

J = 1/(2*m) * sum(h_minus_y .^ 2) + lambda/(2*m) * sum(theta(2:end,:) .^ 2);

% WORKS 1
%grad(1) = 1/m * sum(h_minus_y .* X(:,1));
%grad(2) = 1/m * sum(h_minus_y .* X(:,2)) + lambda/m * theta(2);

% WORKS 2
%grad(1) = 1/m * h_minus_y' * X(:,1);
%grad(2) = 1/m * h_minus_y' * X(:,2) + lambda/m * theta(2);

% WORKS 3
%grad(1) = 1/m * h_minus_y' * X(:,1);
%grad(2) = 1/m * h_minus_y' * X(:,2);
grad = 1/m .* (h_minus_y' * X)';
grad(2:end) = grad(2:end) + lambda/m * theta(2:end);

% =========================================================================

grad = grad(:);

end
