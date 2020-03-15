function [X_train, y_train, X_val, y_val, X_test, y_test, vocabList] = getAllData()

fprintf('\nPreprocessing spamassassin email\n');


if (exist('X_train.mat', 'file') && ...
    exist('y_train.mat', 'file') && ...
    exist('X_val.mat', 'file') && ...
    exist('y_val.mat', 'file') && ...
    exist('X_test.mat', 'file') && ...
    exist('y_test.mat', 'file') && ...
    exist('vocabList.mat', 'file'))
    fprintf('\nload all data\n');
    load('X_train.mat')
    load('y_train.mat')
    load('X_val.mat')
    load('y_val.mat')
    load('X_test.mat')
    load('y_test.mat')
    load('vocabList.mat')
    return;
endif

if (exist('Spams_X_train.mat', 'file') && ...
    exist('Spams_y_train.mat', 'file') && ...
    exist('Spams_X_val.mat', 'file') && ...
    exist('Spams_y_val.mat', 'file') && ...
    exist('Spams_X_test.mat', 'file') && ...
    exist('Spams_y_test.mat', 'file') && ...
    exist('vocabList.mat', 'file'))
    fprintf('\nload all samples\n');
    load('Spams_X_train.mat')
    load('Spams_y_train.mat')
    load('Spams_X_val.mat')
    load('Spams_y_val.mat')
    load('Spams_X_test.mat')
    load('Spams_y_test.mat')
    load('vocabList.mat')
else
    % 1 load all, 2 load lite
    [SpamsOrigin, NonSpamsOrigin] = loadOrConvertEmails(1); 
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

    % print size
    size_Spams_X_train = size(Spams_X_train)
    size_Spams_y_train = size(Spams_y_train)
    size_Spams_X_val = size(Spams_X_val)
    size_Spams_y_val = size(Spams_y_val)
    size_Spams_X_test = size(Spams_X_test)
    size_Spams_y_test = size(Spams_y_test)

    [vocabList, vocabTimes] = createVocabList(Spams_X_train);
    save 'Spams_X_train.mat' Spams_X_train;
    save 'Spams_y_train.mat' Spams_y_train;
    save 'Spams_X_val.mat' Spams_X_val;
    save 'Spams_y_val.mat' Spams_y_val;
    save 'Spams_X_test.mat' Spams_X_test;
    save 'Spams_y_test.mat' Spams_y_test;
    save 'vocabList.mat' vocabList
    save 'vocabTimes.mat' vocabTimes
    size_vocabList = size(vocabList)
    size_vocabTimes = size(vocabTimes)

    filename = "vocabListStatistics.txt";
    fid = fopen(filename, "w");
    for i=1:length(vocabList)
        fputs(fid, sprintf("%d\t%d\t%s\n", i, vocabTimes(i), vocabList{i}));
    endfor
    fclose(fid);
endif

end
