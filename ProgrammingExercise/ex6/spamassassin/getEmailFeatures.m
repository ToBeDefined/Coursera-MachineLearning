function [EmailFeatures] = getEmailFeatures(EmailContents, vocabList)

EmailFeatures = zeros(size(EmailContents, 1), length(vocabList));

fprintf('\ngetEmailFeatures Email Counts: %d\n', size(EmailContents, 1));
for i = 1:size(EmailContents, 1)
    word_indices = []; % row vector
    fprintf('>> getEmailFeatures in Email: %d\n', i);
    word_list = collectWords(EmailContents(i,:));
    for j = 1:length(word_list)
        word = word_list{j};
        idx = find(strcmp(word, vocabList), 1); 
        if ~isempty(idx)
            word_indices = [word_indices, idx]; % row vector
        endif
    endfor
    EmailFeatures(i, word_indices) = 1;
endfor

end
