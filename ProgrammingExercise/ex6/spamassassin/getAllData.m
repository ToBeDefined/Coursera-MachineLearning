function [X_train, y_train, X_val, y_val, X_test, y_test, vocabList] = getAllData()

X_train = [];
y_train = [];
X_val = [];
y_val = [];
X_test = [];
y_test = [];
vocabList = {};

% Use Custom Saver because use Octave's save command always failed 
% (Email origin data too large)
addpath('BigDataSaver');

fprintf('\nPreprocessing spamassassin email\n');


[X_train, load_X_train_success] = loadBigData('X_train.mat');
[X_val,   load_X_val_success]   = loadBigData('X_val.mat');
[X_test,  load_X_test_success]  = loadBigData('X_test.mat');
[y_train, load_y_train_success] = loadBigData('y_train.mat');
[y_val,   load_y_val_success]   = loadBigData('y_val.mat');
[y_test,  load_y_test_success]  = loadBigData('y_test.mat');

if  load_X_train_success && ...
    load_X_val_success && ...
    load_X_test_success && ...
    load_y_train_success && ...
    load_y_val_success && ...
    load_y_test_success && ...
    exist('vocabListStatistics.txt', 'file')
    vocabList = getVocabList();
    return;
endif



%====== Split Spams&NonSpams to Spams_X_train Spams_X_val Spams_X_test Spams_y_train Spams_y_val Spams_y_test
isResplit = 0;
[Spams_X_train, load_Spams_X_train_success] = loadBigData('Spams_X_train.mat');
[Spams_X_val,   load_Spams_X_val_success]   = loadBigData('Spams_X_val.mat');
[Spams_X_test,  load_Spams_X_test_success]  = loadBigData('Spams_X_test.mat');
[Spams_y_train, load_Spams_y_train_success] = loadBigData('Spams_y_train.mat');
[Spams_y_val,   load_Spams_y_val_success]   = loadBigData('Spams_y_val.mat');
[Spams_y_test,  load_Spams_y_test_success]  = loadBigData('Spams_y_test.mat');

if  ~load_Spams_X_train_success || ...
    ~load_Spams_X_val_success || ...
    ~load_Spams_X_test_success || ...
    ~load_Spams_y_train_success || ...
    ~load_Spams_y_val_success || ...
    ~load_Spams_y_test_success

    fprintf('\nload and ramdom split SpamsEmail&NonSpamEmail\n');
    
    [SpamsOrigin, NonSpamsOrigin] = loadOrConvertEmails(); 
    Spams = [SpamsOrigin; NonSpamsOrigin];
    yVec = [ones(size(SpamsOrigin, 1), 1); zeros(size(NonSpamsOrigin, 1), 1)];

    isResplit = 1;
    delete 'vocabListStatistics.txt' % vocabListStatistics.txt is related by Spams_X_train

    % random 60% for training
    count = length(yVec);
    train_count = int64(count * 0.6);
    train_index = randperm(count, train_count);
    Spams_X_train = Spams(train_index, :);
    Spams_y_train = yVec(train_index, :);

    % remove training data
    Spams(train_index, :) = [];
    yVec(train_index, :) = [];

    % random 20% for validation 
    count = length(yVec);
    val_count = int64(count * 0.5);
    val_index = randperm(count, val_count);
    Spams_X_val = Spams(val_index, :);
    Spams_y_val = yVec(val_index, :);

    % remove validation data
    Spams(val_index, :) = [];
    yVec(val_index, :) = [];

    % last 20% for test, random rows index
    count = length(yVec);
    test_count = count;
    test_index = randperm(count, test_count);
    Spams_X_test = Spams(test_index, :);
    Spams_y_test = yVec(test_index, :);

    saveBigData('Spams_X_train.mat', Spams_X_train);
    saveBigData('Spams_X_val.mat',   Spams_X_val);
    saveBigData('Spams_X_test.mat',  Spams_X_test);
    saveBigData('Spams_y_train.mat', Spams_y_train);
    saveBigData('Spams_y_val.mat',   Spams_y_val);
    saveBigData('Spams_y_test.mat',  Spams_y_test);
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
if isResplit || isempty(X_train)
    X_train = getEmailFeatures(Spams_X_train, vocabList);
    saveBigData('X_train.mat', X_train);
endif

if isResplit || isempty(X_val)
    X_val = getEmailFeatures(Spams_X_val, vocabList);
    saveBigData('X_val.mat', X_val);
endif

if isResplit || isempty(X_test)
    X_test = getEmailFeatures(Spams_X_test, vocabList);
    saveBigData('X_test.mat', X_test);
endif

y_train = Spams_y_train;
saveBigData('y_train.mat', y_train);

y_val   = Spams_y_val;
saveBigData('y_val.mat', y_val);

y_test  = Spams_y_test;
saveBigData('y_test.mat', y_test);

end
