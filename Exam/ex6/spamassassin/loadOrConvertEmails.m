function [Spams, NonSpams] = loadOrConvertEmails()

Spams = [];
NonSpams = [];

if exist('EmailContent.mat', 'file')
    fprintf('\nload EmailContent.mat\n');
    % Spams NonSpams
    load 'EmailContent.mat'
    return;
endif

SpamFileContents = [];
NonSpamFileContents = [];

% load/convert spam email
if exist('spamFileContents.mat', 'file')
    fprintf('\nload spamFileContents.mat\n');
    load 'spamFileContents.mat'
else
    fprintf('\nPreprocessing convert spamassassin email (spam)\n');
    SpamFileContents = [];
    SpamDir = './spam/';
    spamFileInfo = dir([SpamDir '*.bz2']);

    for i = 1:length(spamFileInfo)
        zipFile = [SpamDir spamFileInfo(i).name];
        ZipEmailContents = readZipFile(zipFile);
        SpamFileContents = [SpamFileContents; ZipEmailContents];
    endfor
    save 'spamFileContents.mat' SpamFileContents
endif

% load/convert non-spam email
if exist('nonSpamFileContents.mat', 'file')
    fprintf('\nload nonSpamFileContents.mat\n');
    load 'nonSpamFileContents.mat'
else
    fprintf('\nPreprocessing convert spamassassin email (non-spam)\n');
    NonSpamFileContents = [];
    NonSpamDir = './ham/';
    nonSpamFileInfo = dir([NonSpamDir '*.bz2']);

    for i = 1:length(nonSpamFileInfo)
        zipFile = [NonSpamDir nonSpamFileInfo(i).name];
        ZipEmailContents = readZipFile(zipFile);
        NonSpamFileContents = [NonSpamFileContents; ZipEmailContents];
    endfor
    save 'nonSpamFileContents.mat' NonSpamFileContents
endif

Spams = SpamFileContents;
NonSpams = NonSpamFileContents;

save 'EmailContent.mat' Spams NonSpams
delete 'spamFileContents.mat' 'nonSpamFileContents.mat'

end
