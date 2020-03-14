function [Spams, NonSpams] = loadOrConvertEmails(tag)

Spams = [];
NonSpams = [];
SpamFileContents = [];
NonSpamFileContents = [];
SpamFileContentsLite = [];
NonSpamFileContentsLite = [];

% load/convert spam email
loadSpamFileName = 'spamFileContents.mat';
loadNonSpamFileName = 'nonSpamFileContents.mat';
if tag != 1
    loadSpamFileName = 'spamFileContentsLite.mat';
    loadNonSpamFileName = 'nonSpamFileContentsLite.mat';
endif

if exist(loadSpamFileName, 'file')
    fprintf('\nload %s\n', loadSpamFileName);
    load(loadSpamFileName);
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
    save 'spamFileContents.mat' SpamFileContents;

    SpamFileContentsLite = randomSample(SpamFileContents, 1000);
    save 'spamFileContentsLite.mat' SpamFileContentsLite;
endif

% load/convert non-spam email
if exist(loadNonSpamFileName, 'file')
    fprintf('\nload %s\n', loadNonSpamFileName);
    load(loadNonSpamFileName);
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
    save 'nonSpamFileContents.mat' NonSpamFileContents;

    NonSpamFileContentsLite = randomSample(NonSpamFileContents, 1000);
    save 'nonSpamFileContentsLite.mat' NonSpamFileContentsLite;
endif

if tag == 1
    Spams = SpamFileContents;
    NonSpams = NonSpamFileContents;
else
    Spams = SpamFileContentsLite;
    NonSpams = NonSpamFileContentsLite;
endif

end
