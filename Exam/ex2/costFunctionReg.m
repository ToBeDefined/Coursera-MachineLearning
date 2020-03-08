function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% calculate J
z = X * theta;
h = sigmoid(z);
eachCost = (-y)' * log(h) - (ones(m, 1) - y)' * log(ones(m, 1) - h);
eachCostAddition = lambda/(2 * m) .* theta(2:end) .^ 2;
J = 1/m * sum(eachCost) + sum(eachCostAddition);

% calculate grad vector
grad = X' * (h - y) ./ m;
regularize_addition = (lambda / m) .* [0; ones(length(theta)-1, 1)];
regularize_addition = regularize_addition .* theta;
grad = grad + regularize_addition;




% =============================================================

end
