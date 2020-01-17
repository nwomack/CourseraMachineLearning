function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

alpha_over_m = alpha/m;

%num_iters = 10

%fprintf("theta = %f\n", theta);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    T = zeros(length(theta),1);

    for i=1:m
        h = X(i,:) * theta;
        T = T + (h - y(i)) * X(i, :)';
    end

    theta = theta - alpha_over_m * T;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    %fprintf("J_history = %x\n", J_history(iter));

end

end
