function [BigDataVariable, success] = loadBigData(loadFileName)

BigDataVariable = [];
success = 0;

isValidFileName = size(loadFileName, 1);
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
    componentFileName = sprintf('%s_%d', loadFileName, fileIdx);
    if ~exist(fullfile(basePath, dataDir, componentFileName), 'file')
        return
    endif

    while exist(fullfile(basePath, dataDir, componentFileName), 'file')
        fprintf('\nloading %s...', loadFileName);
        load(fullfile(basePath, dataDir, componentFileName));
        
        % last component file saved var is 'success', read success, then break loop
        if success
            break;
        endif
        if exist('BigDataComponent', 'var') 
            % is have 'BigDataComponent' variable
            if fileIdx == 1
                BigDataVariable = BigDataComponent;
            else
                BigDataVariable = [BigDataVariable; BigDataComponent];
            endif
            % clear 'BigDataComponent' variable
            clear BigDataComponent;
        endif
        
        fileIdx = fileIdx + 1;
        componentFileName = sprintf('%s_%d', loadFileName, fileIdx);
    endwhile
catch
    fprintf('\nload %s error.\n', loadFileName);
    return
end % end try

if success
    fprintf('\nload %s success.\n', loadFileName);
else
    fprintf('\nload %s error.\n', loadFileName);
endif

end
