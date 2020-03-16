function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%% ------------------------------ Add ones to X get A1, and calculate A2
m = size(X, 1);
A1 = [ones(m, 1), X];
% sizeA1 = size(A1)
% sizeTheta1Trans = size(Theta1')
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);

%% ------------------------------ Add ones to A2 get new A2, and calculate A3
m = size(A2, 1);
A2 = [ones(m, 1), A2];
% sizeA2 = size(A2)
% sizeTheta2Trans = size(Theta2')
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

%% ------------------------------ get A3 max column index of each row, return p
% sizeA3 = size(A3)
[maxNum, idx] = max(A3, [], 2);
p = idx;



% =========================================================================


end
