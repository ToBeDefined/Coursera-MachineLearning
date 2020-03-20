
%% ========================= Initialization & load data ========================= 
clear ; close all; clc

addpath('libsvm');

[X_train, y_train, X_val, y_val, X_test, y_test, ~] = getAllData();

clc;

fprintf('\nLoad data success, Press enter to continue.\n');
pause;
clc;

% >> svmtrain
% Usage: model = svmtrain(training_label_vector, training_instance_matrix, 'libsvm_options');
% libsvm_options:
% -s svm_type : set type of SVM (default 0)
%   0 -- C-SVC    (multi-class classification)
%   1 -- nu-SVC   (multi-class classification)
%   2 -- one-class SVM
%   3 -- epsilon-SVR  (regression)
%   4 -- nu-SVR   (regression)
% -t kernel_type : set type of kernel function (default 2)
%   0 -- linear: u'*v
%   1 -- polynomial: (gamma*u'*v + coef0)^degree
%   2 -- radial basis function: exp(-gamma*|u-v|^2)
%   3 -- sigmoid: tanh(gamma*u'*v + coef0)
%   4 -- precomputed kernel (kernel values in training_instance_matrix)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
% -v n: n-fold cross validation mode
% -q : quiet mode (no outputs)

% >> svmpredict
% Usage: [predicted_label, accuracy, decision_values/prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')
%        [predicted_label] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')
% Parameters:
%   model: SVM model structure from svmtrain.
%   libsvm_options:
%     -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet
%     -q : quiet mode (no outputs)
% Returns:
%   predicted_label: SVM prediction output vector.
%   accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.
%   prob_estimates: If selected, probability estimate vector.



%% ========================= Training with different C ========================= 
CVec = [[0.01:0.02:0.09]'; [0.1:0.2:0.9]'; [1:2:9]';];
Accuracy = zeros(1:length(CVec), 3);
Models = {};
for i = 1:length(CVec)
    C = CVec(i);
    fprintf('\n>> training %d: C = %d...\n', i, C);
    model = ...
        svmtrain(y_train, X_train, sprintf('-s 0 -h 0 -t 0 -c %d', C));
    [predicted_label, accuracy, prob_estimates] = ...
        svmpredict(y_val, X_val, model);
    Accuracy(i,:) = accuracy';
    Models{i} = model;
endfor

fprintf('\n\n>>>> Training Models Validation with Different C:\n\n');
fprintf('CVec\t\tAccuracy\tMeanSquaredError\tSquaredCorrelationCoefficient\n');
% disp([CVec Accuracy]);
for i=1:length(CVec)
    fprintf('%f\t%f\t%f\t\t%f\n', CVec(i), Accuracy(i,1), Accuracy(i,2), Accuracy(i,3));
endfor
[maxA, maxIdx] = max(Accuracy(:,1), [], 1);
C = CVec(maxIdx);
model = Models{maxIdx};
fprintf('\n\n>>>> Max Accuracy = %f, C = %d\n', maxA, C);

fprintf('\nPress enter to continue use training set and validation set retraining model.\n');
pause;

clc;


%% ================== Retraining use C for training set and validation set ==================
fprintf('\n>> retraining model use training set and validation set...\n');
retraining_model = ...
    svmtrain([y_train; y_val], [X_train; X_val], sprintf('-s 0 -h 0 -t 0 -c %d', C));

fprintf('\n>> predict for origin model(only training set trains model)...\n');
[origin_model_predicted_label, origin_model_accuracy, origin_model_prob_estimates] = ...
    svmpredict(y_test, X_test, model);

fprintf('\n>> predict for retraining model(use training set and validation set trains model)...\n');
[retrain_model_predicted_label, retrain_model_accuracy, retrain_model_prob_estimates] = ...
    svmpredict(y_test, X_test, retraining_model);

fprintf('\n\n>>>> Training Models Results for Test Set:\n\n');
fprintf('TrainingSet\t\tAccuracy\tMeanSquaredError\tSquaredCorrelationCoefficient\n');
fprintf('---------------------------------------------------------------------------------------------\n');
fprintf('%s\t\t%f\t%f\t\t%f\n', 'Only TrainSet', origin_model_accuracy');
fprintf('%s\t%f\t%f\t\t%f\n', 'TrainSet & ValSet', retrain_model_accuracy');

