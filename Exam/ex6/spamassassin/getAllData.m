function [X_train, y_train, X_val, y_val, X_test, y_test, vocabList] = getAllData()

X_train = [];
y_train = [];
X_val = [];
y_val = [];
X_test = [];
y_test = [];
vocabList = {};

fprintf('\nPreprocessing spamassassin email\n');

% if have 'EmailSplitFeatures.mat' and 'vocabListStatistics.txt' file, load and return
if exist('EmailSplitFeatures.mat', 'file') && exist('vocabListStatistics.txt', 'file')
    fprintf('\nload all data\n');
    % X_train X_val X_test y_train y_val y_test
    load 'EmailSplitFeatures.mat'
    vocabList = getVocabList();
    return;
endif

%====== Split Spams&NonSpams to Spams_X_train Spams_X_val Spams_X_test Spams_y_train Spams_y_val Spams_y_test
isResplit = 0;
if exist('EmailSplitContents.mat', 'file')
    fprintf('\nload EmailSplitContents.mat\n');
    % Spams_X_train Spams_X_val Spams_X_test Spams_y_train Spams_y_val Spams_y_test
    load 'EmailSplitContents.mat'
else
    % 1 load all, 2 load lite
    [SpamsOrigin, NonSpamsOrigin] = loadOrConvertEmails(); 
    Spams = [SpamsOrigin; NonSpamsOrigin];
    yVec = [ones(size(SpamsOrigin, 1), 1); zeros(size(NonSpamsOrigin, 1), 1)];

    size_Spams = size(Spams)
    size_yVec = size(yVec)

    % random 60% for training
    count = length(yVec);
    train_count = int64(count * 0.6);
    train_index = randperm(count, train_count);
    Spams_X_train = Spams(train_index, :);
    Spams_y_train = yVec(train_index, :);

    % remove training data
    Spams(train_index, :) = [];
    yVec(train_index, :) = [];
    count = length(yVec);

    % random 20% for validation 
    val_count = int64(count * 0.5);
    val_index = randperm(count, val_count);
    Spams_X_val = Spams(val_index, :);
    Spams_y_val = yVec(val_index, :);

    % remove validation data
    Spams(val_index, :) = [];
    yVec(val_index, :) = [];
    count = length(yVec);

    % last 20% for test, random rows index
    test_count = count;
    test_index = randperm(count, test_count);
    Spams_X_test = Spams(test_index, :);
    Spams_y_test = yVec(test_index, :);

    save 'EmailSplitContents.mat' Spams_X_train Spams_X_val Spams_X_test Spams_y_train Spams_y_val Spams_y_test
    isResplit = 1;
    delete 'vocabListStatistics.txt'
endif

% create 'vocabListStatistics.txt'
if isResplit || ~exist('vocabListStatistics.txt', 'file')
    [vocabList, vocabTimes] = createVocabList(Spams_X_train);
    % save all words
    filename = "vocabListStatistics.txt";
    fid = fopen(filename, "w");
    for i=1:length(vocabList)
        fputs(fid, sprintf("%d\t%d\t%s\n", i, vocabTimes(i), vocabList{i}));
    endfor
    fclose(fid);
endif

% reload vocabList
vocabList = getVocabList();

% in here, must don't have file: 'EmailSplitFeatures.mat'
% convet Spams to X
if isResplit || ~exist('X_train.mat', 'file')
    X_train = getEmailFeatures(Spams_X_train, vocabList);
    save -v7 'X_train.mat' X_train
else
    load 'X_train.mat'
endif

if isResplit || ~exist('X_val.mat', 'file')
    X_val = getEmailFeatures(Spams_X_val, vocabList);
    save -v7 'X_val.mat' X_val
else 
    load 'X_val.mat'
endif

if isResplit || ~exist('X_test.mat', 'file')
    X_test = getEmailFeatures(Spams_X_test, vocabList);
    save -v7 'X_test.mat' X_test
else
    load 'X_test.mat'
endif

y_train = Spams_y_train;
y_val   = Spams_y_val;
y_test  = Spams_y_test;

save -v7 'EmailSplitFeatures.mat' X_train X_val X_test y_train y_val y_test
delete 'X_train.mat' 'X_val.mat' 'X_test.mat'

end
