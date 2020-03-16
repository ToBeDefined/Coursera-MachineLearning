function [Spams, NonSpams] = loadOrConvertEmails()

Spams = [];
NonSpams = [];

% Use Custom Saver because use Octave's save command always failed 
% (Email origin data too large)
addpath('BigDataSaver');

[Spams, loadSpamsSuccess] = loadBigData('Spams.mat');
[NonSpams, loadNonSpamsSuccess] = loadBigData('NonSpams.mat');

% load/convert spam email
if ~loadSpamsSuccess
    fprintf('\nPreprocessing convert spamassassin email (spam)\n');
    Spams = [];
    SpamDir = './spam/';
    spamFileInfo = dir([SpamDir '*.bz2']);

    for i = 1:length(spamFileInfo)
        zipFile = [SpamDir spamFileInfo(i).name];
        ZipEmailContents = readZipFile(zipFile);
        Spams = [Spams; ZipEmailContents];
    endfor
    saveBigData('Spams.mat', Spams);
endif

% load/convert non-spam email
if ~loadNonSpamsSuccess
    fprintf('\nPreprocessing convert spamassassin email (non-spam)\n');
    NonSpams = [];
    NonSpamDir = './ham/';
    nonSpamFileInfo = dir([NonSpamDir '*.bz2']);

    for i = 1:length(nonSpamFileInfo)
        zipFile = [NonSpamDir nonSpamFileInfo(i).name];
        ZipEmailContents = readZipFile(zipFile);
        NonSpams = [NonSpams; ZipEmailContents];
    endfor
    saveBigData('NonSpams.mat', NonSpams);
endif

end
