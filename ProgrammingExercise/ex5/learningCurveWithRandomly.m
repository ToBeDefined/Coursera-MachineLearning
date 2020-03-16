function [error_train, error_val] = ...
    learningCurveWithRandomly(X, y, Xval, yval, lambda)

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

training_times = 50;

% different training set sizes
for sample_size = 1:m
    for trainingTime = 1:training_times
        % get random training set and cross validation set
        [TrainX, trainY] = randomSample(X, y, sample_size);
        [ValidationX, validationY] = randomSample(Xval, yval, sample_size);
        % traning theta with i training set size
        theta = trainLinearReg(TrainX, trainY, lambda);
        % compute the error on the training sets and cross validation sets
        error_train(sample_size) = error_train(sample_size) + linearRegCostFunction(TrainX, trainY, theta, 0);
        error_val(sample_size) = error_val(sample_size) + linearRegCostFunction(ValidationX, validationY, theta, 0);
    endfor
endfor

% get mean value for training times
error_train = error_train ./ training_times;
error_val = error_val ./ training_times;

end
