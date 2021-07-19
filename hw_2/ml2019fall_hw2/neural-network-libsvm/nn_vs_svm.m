load('digit_data', 'X', 'y');
load('weights', 'Theta1', 'Theta2');

p = feedforward(Theta1, Theta2, X);
fprintf('Error rate for NN is %f.\n', length(find(p ~= y))/length(p));

train_X = X(:, 1:2500);
train_y = y(1:2500);
test_X = X(:, 2501:end);
test_y = y(2501:end);

% YOUR CODE HERE
% Trainning and testing using one-vs-all with LIBLINEAR
