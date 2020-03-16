function [BigDataVariable, success] = loadBigData(loadFileName)

BigDataVariable = [];
success = 0;

isValidFileName = size(loadFileName, 1);
if ~isValidFileName
    return
endif

if ~exist('BigData', 'dir')
    return
endif


try

cd BigData

fileIdx = 1;
componentFileName = sprintf('%s_%d', loadFileName, fileIdx);
if ~exist(componentFileName, 'file')
    return
endif

while exist(componentFileName, 'file')
    fprintf('\nloading %s...', loadFileName);
    load(componentFileName);
    
    % last component file saved var is 'success', not 'BigDataComponent'
    if exist('BigDataComponent', 'var') 
        if fileIdx == 1
            BigDataVariable = BigDataComponent;
        else
            BigDataVariable = [BigDataVariable; BigDataComponent];
        endif
        clear BigDataComponent;
    endif
    
    fileIdx = fileIdx + 1;
    componentFileName = sprintf('%s_%d', loadFileName, fileIdx);
endwhile

catch
    fprintf('\nload %s error.\n', loadFileName);
    cd ..
    return
end % end try

if success
    fprintf('\nload %s success.\n', loadFileName);
else
    fprintf('\nload %s error.\n', loadFileName);
endif

cd ..

end
