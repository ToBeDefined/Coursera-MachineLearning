function removeBigData(removeFileName)

isValidFileName = size(removeFileName, 1);
if ~isValidFileName
    return
endif

if ~exist('BigData', 'dir')
    return
endif


try

cd BigData

fileIdx = 1;
componentFileName = sprintf('%s_%d', removeFileName, fileIdx);
while exist(componentFileName, 'file')
    fprintf('\nremoving %s...', removeFileName);
    delete(componentFileName);
    fileIdx = fileIdx + 1;
    componentFileName = sprintf('%s_%d', removeFileName, fileIdx);
endwhile

catch
    fprintf('\nremove %s error.\n', removeFileName);
    cd ..
    return
end % end try

fprintf('\nremove %s success.\n', removeFileName);

cd ..

end
% [myBigData2, isReadSuccess] = loadBigData('myBigData.mat');
