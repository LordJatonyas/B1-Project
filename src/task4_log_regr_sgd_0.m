clear all;
close all;
clc;


% Note: Remember to Change class labels {1,2} to values for y appropriate for log-regr => y={0,1}

rng(12345) % Random number generator seed. Comment it out to use random seed.

% Create training/val data
n_samples = 1000;
[X_train_val, regression_targets_train_val, class_labels_train_val] = create_data(n_samples);
regression_targets_train_val = NaN; % won't be used for classification
y_train_val = (class_labels_train_val==1)*(0) + (class_labels_train_val==2)*(1);
X_train_val = cat(2, ones(n_samples,1), X_train_val); % concat 1 for bias
%...

% Divide into Training and Validation sets in a 80-20 split
[X_train, y_train, X_val, y_val] = divide_data(X_train_val, y_train_val, n_samples);

% Create Test set of 20000 datapoints
n_test = 20000;
[X_test, regression_test, class_labels_test] = create_data(n_test);
regression_test = NaN; % won't be used for classification
y_test = (class_labels_test==1)*(0) + (class_labels_test==2)*(1);
X_test = cat(2, ones(n_test,1), X_test); % concat 1 for bias
%....

% Optimize - Logistic Regression - Stochastic Gradient Descent
n_iters = 1000;
lambda = 1;
batch_sizes = [1 10 20 50 100];
seeds = 1:20;
thetas = zeros(size(X_train, 2), length(batch_sizes));
mean_loglosses = zeros(length(batch_sizes), n_iters);
train_errors = zeros(length(seeds), length(batch_sizes));
valid_errors = zeros(length(seeds), length(batch_sizes));
test_errors = zeros(length(seeds), length(batch_sizes));

for i = 1:length(batch_sizes)
    for j = 1:length(seeds)
        rng(seeds(j));

        % Create training/val data
        n_samples = 1000;
        [X_train_val, regression_targets_train_val, class_labels_train_val] = create_data(n_samples);
        regression_targets_train_val = NaN; % won't be used for classification
        y_train_val = (class_labels_train_val==1)*(0) + (class_labels_train_val==2)*(1);
        X_train_val = cat(2, ones(n_samples,1), X_train_val); % concat 1 for bias
        
        % Create Test set of 20000 datapoints
        n_test = 20000;
        [X_test, regression_test, class_labels_test] = create_data(n_test);
        regression_test = NaN; % won't be used for classification
        y_test = (class_labels_test==1)*(0) + (class_labels_test==2)*(1);
        X_test = cat(2, ones(n_test,1), X_test); % concat 1 for bias

        % Analysis
        [theta_opt, mean_lls] = logistic_regression_sgd(X_train, y_train, batch_sizes(i), lambda, n_iters);
        thetas(:, i) = theta_opt;
        mean_loglosses(i, :) = mean_lls;
        train_errors(j, i) = classif_error(y_train, log_regr(X_train, theta_opt));
        valid_errors(j, i) = classif_error(y_val, log_regr(X_val, theta_opt));
        test_errors(j, i) = classif_error(y_test, log_regr(X_test, theta_opt));
    end
end

means_train = mean(train_errors, 1);
stds_train = std(train_errors, 1);
means_and_stds_train = cat(1, means_train, stds_train);
fprintf("Experiment Train Error Means:\n");
disp(means_and_stds_train(1,:));
fprintf("Experiment Train Error Standard Deviations:\n");
disp(means_and_stds_train(2,:));

means_val = mean(valid_errors, 1);
stds_val = std(valid_errors, 1);
means_and_stds_val = cat(1, means_val, stds_val);
fprintf("Experiment Validation Error Means:\n");
disp(means_and_stds_val(1,:));
fprintf("Experiment Validation Error Standard Deviations:\n");
disp(means_and_stds_val(2,:));

means_test = mean(test_errors, 1);
stds_test = std(test_errors, 1);
means_and_stds_test = cat(1, means_test, stds_test);

plot(mean_loglosses(5, :));
title("Mean Log-loss against Number of Iterations");
ylabel("Mean Log-loss");
xlabel("Number of Iterations");
saveas(gcf, "../report/task4-mean_loglosses.png");

fprintf("Validation Error: %.4f +/- %.4f\n", means_and_stds_val(1,length(batch_sizes)),means_and_stds_val(2,length(batch_sizes)));
fprintf("Test Error: %.4f +/- %.4f\n", means_and_stds_test(1,length(batch_sizes)),means_and_stds_test(2,length(batch_sizes)));

disp(classif_error(y_test, log_regr(X_test, theta_opt)));
disp(classif_error(y_val, log_regr(X_val, theta_opt)));
% ...



function [theta_opt, mean_lls] = logistic_regression_sgd(X_train, y_train, batch_size, learning_rate, iters_total)
    n_features = size(X_train, 2);
    theta_curr = zeros(n_features, 1);
    mean_lls = zeros(1, iters_total);
    for i = 1:iters_total
        % Batching
        indices = randperm(length(X_train));
        X_b = X_train(indices(1:batch_size), :);
        y_b = y_train(indices(1:batch_size), :);
        % predicted y
        y_pred = log_regr(X_b, theta_curr);
        % grad of log-loss
        grad = X_b' * (y_pred - y_b);
        % update theta
        theta_curr = theta_curr - learning_rate * 1 / length(X_b) * grad;
        mean_lls(1, i) = mean_logloss(X_train, y_train, theta_curr);
    end
    theta_opt = theta_curr;
end

function mean_logloss = mean_logloss(X, y_real, theta)
    y_pred = log_regr(X, theta);
    logloss = sum(-y_real .* log(y_pred) - (ones(size(y_real))-y_real) .* log(ones(size(y_pred))-y_pred));
    mean_logloss = 1/length(X) * logloss;
end


function y_pred = log_regr(X, theta)
    z = X * theta;
    y_pred = ones(size(z)) ./ (ones(size(z)) + exp(-z));
end

function err_perc = classif_error(y_real, y_pred)
    wrong_count = sum(y_real ~= round(y_pred));
    err_perc = wrong_count / length(y_pred);
end

% Training-Validation Split
function [X_train, y_train, X_val, y_val] = divide_data(X, y, n)
    indices = randperm(n);
    test_cap = n * 0.8;
    X_train = X(indices(1:test_cap), :);
    y_train = y(indices(1:test_cap), 1);
    X_val = X(indices(test_cap+1:n), :);
    y_val = y(indices(test_cap+1:n), 1);
end