function removeBigData(removeFileName)

isValidFileName = size(removeFileName, 1);
if ~isValidFileName
    return
endif

basePath = pwd;
dataDir = 'BigData';
if ~exist(dataDir, 'dir')
    return
endif

try
    fileIdx = 1;
    hasRemoveFile = 0;
    componentFileName = sprintf('%s_%d', removeFileName, fileIdx);
    while exist(fullfile(basePath, dataDir, componentFileName), 'file')
        hasRemoveFile = 1;
        fprintf('\nremoving %s...', removeFileName);
        delete(fullfile(basePath, dataDir, componentFileName));
        fileIdx = fileIdx + 1;
        componentFileName = sprintf('%s_%d', removeFileName, fileIdx);
    endwhile
catch
    fprintf('\nremove %s error.\n', removeFileName);
    return
end % end try

if hasRemoveFile
    fprintf('\nremove %s success.\n', removeFileName);
else
    fprintf('\nno old file of %s.\n', removeFileName);
endif

end
% [myBigData2, isReadSuccess] = loadBigData('myBigData.mat');
