function [EmailContents] = readZipFile(zipFile)

EmailContents = [];

fprintf('>>  %s\n', zipFile);
system("rm -rf ./temp/");
untar(zipFile, "./temp/");
dataDir = dir("./temp/");
for i = 1:length(dataDir)
    if isequal(dataDir(i).name,'.') || ...
        isequal(dataDir(i).name,'..') || ...
        ~dataDir(i).isdir
        continue;
    end
    dataFile = dir(["./temp/" dataDir(i).name]);
    for j = 1:length(dataFile)
        if isequal(dataFile(j).name,'.') || ...
            isequal(dataFile(j).name,'..') || ...
            isequal(dataFile(j).name,'cmds')
            continue;
        end
        filePath = fullfile("./temp/", dataDir(i).name, dataFile(j).name);
        fprintf('>> >>  %s\n', filePath);
        fileContent = readFile(filePath);
        % remove email headers
        email_contents = regexprep(fileContent, '.*?\n\n', '', 'once');
        email_contents = preProcessEmail(email_contents);
        EmailContents = [EmailContents; email_contents];
    end
endfor

end
