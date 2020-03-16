function success = saveBigData(saveFileName, BigDataVariable)

success = 0;
isValidFileName = size(saveFileName, 1);
if ~isValidFileName
    return
endif

rowCount = size(BigDataVariable, 1);
if rowCount == 0
    return
endif

beginIdx = 1;
fileIdx = 1;
oneRow = BigDataVariable(1,:);
oneRowBytes = whos('oneRow').bytes;
% set maximum 2GB data in one file
rowsOfOneFile = int64(floor(2000000000 / oneRowBytes));

isSaveAllData = 0;

basePath = pwd;
dataDir = 'BigData';

if ~exist(dataDir, 'dir')
    mkdir(dataDir);
endif

% remove old data files
fprintf('\nremoving old data files: %s...', saveFileName);
removeBigData(saveFileName)

try
    while ~isSaveAllData
        % all data is saved, break while loop
        if (beginIdx > rowCount)
            isSaveAllData = 1;
            break;
        endif

        % 'rowsOfOneFile' rows in one data file
        endIdx = beginIdx + rowsOfOneFile - 1;

        % more than 'rowCount' rows
        if (endIdx > rowCount)
            endIdx = rowCount;
        endif

        % save data component
        BigDataComponent = BigDataVariable(beginIdx:endIdx, :);
        saveComponentFileName = sprintf('%s_%d', saveFileName, fileIdx);
        fprintf('\nsaving %s...', saveFileName);
        save(fullfile(basePath, dataDir, saveComponentFileName), 'BigDataComponent', '-v7');

        beginIdx = endIdx + 1;
        fileIdx = fileIdx + 1;
    endwhile

    success = isSaveAllData;
    % fileIdx is already added in the while loop
    saveTagFileName = sprintf('%s_%d', saveFileName, fileIdx);
    save(fullfile(basePath, dataDir, saveTagFileName), 'success', '-v7');
catch
    fprintf('\nsave %s error.\n', saveFileName);
    return
end % end try

if success
    fprintf('\nsave %s success.\n', saveFileName);
else
    fprintf('\nsave %s error.\n', saveFileName);
endif

end
