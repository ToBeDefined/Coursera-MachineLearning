function vocabList = getVocabList()
fid = fopen('vocabListStatistics.txt');

% only use the top 5000 words
n = 5000;

vocabList = cell(n, 1);
for i = 1:n
    % Word Index (can ignore since it will be = i)
    fscanf(fid, '%d', 1);
    % Word used times (ignore)
    fscanf(fid, '%d', 1);
    % Actual Word
    vocabList{i} = fscanf(fid, '%s', 1);
end

fclose(fid);

end
