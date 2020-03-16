function [X_randoms] = randomSample(X, random_len)

index = randperm(size(X, 1), random_len);
X_randoms = X(index, :);

end
