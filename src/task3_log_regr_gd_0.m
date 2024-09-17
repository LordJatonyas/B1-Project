clear all;
close all;
clc;

% Note: Remember to Change class labels {1,2} to values for y appropriate for log-regr => y={0,1}

rng(12345) % Random number generator seed. Comment it out to use random seed.

% Create training/val data
n_samples = 1000;
[X_train_val, regression_targets_train_val, class_labels_train_val] = create_data(n_samples);
y_train_val = (class_labels_train_val==1)*(0) + (class_labels_train_val==2)*(1); % If class 1, y=0. If class 2, y=1.
regression_targets_train_val = NaN;  % Wont be used for classification.
X_train_val = cat(2, ones(n_samples,1), X_train_val); % concat 1 for bias

% ........
[X_train, y_train, X_val, y_val] = divide_data(X_train_val, y_train_val, n_samples);

% Optimize - Logistic Regression - Gradient Descent
lambdas = [0.00001 0.0001 0.001 0.01 0.1 1];
n_iters = 1000;
errors = zeros(2, 6);
ave_class_loss = zeros(6, n_iters);

for i = 1:length(lambdas)
    lambda = lambdas(i);
    [theta_opt, mean_loglosses] = logistic_regression_gd(X_train, y_train, lambda, n_iters);
    ave_class_loss(i,:) = mean_loglosses;
    errors(1, i) = classif_error(y_train, log_regr(X_train, theta_opt));
    errors(2, i) = classif_error(y_val, log_regr(X_val, theta_opt));
end

plot(ave_class_loss(6,:));
title("Mean Log-loss against Number of Iterations");
ylabel("Mean Log-loss");
xlabel("Number of Iterations");
saveas(gcf, "../report/task3-mean_loglosses.png");

fprintf("1st Experiment Training Classification Error Ratios\n");
disp(errors(1,:));
fprintf("1st Experiment Validation Classification Error Ratios\n");
disp(errors(2,:));

% Create test data
test_samples = 20000;
[X_test, regression_targets_test, class_labels_test] = create_data(test_samples);
y_test = (class_labels_test==1)*(0) + (class_labels_test==2)*(1); % If class 1, y=0. If class 2, y=1.
regression_targets_test = NaN;  % Wont be used for classification.
X_test = cat(2, ones(test_samples,1), X_test); % concat 1 for bias

valid_error = classif_error(y_val, log_regr(X_val, theta_opt));
test_error = classif_error(y_test, log_regr(X_test, theta_opt));
fprintf("Validation Error: %.4f\n", valid_error);
fprintf("Test Error: %.4f\n", test_error);

% Experiment Time
n_iters = 1000;
lambda = 1;
n_samples = [10 20 100 1000 10000];
train_classification_errors = zeros(20, 5);
val_classification_errors = zeros(20, 5);
test_classification_errors = zeros(20, 5);

for i=1:length(n_samples)
    for j=1:20
        rng(j);
        [X, regression_targets, class_labels] = create_data(n_samples(i));
        y = (class_labels==1)*(0) + (class_labels==2)*(1); % If class 1, y=0. If class 2, y=1.
        regression_targets = NaN;  % Wont be used for classification.
        X = cat(2, ones(n_samples(i),1), X); % concat 1 for bias

        test_samples = 20000;
        [X_test, regression_targets_test, class_labels_test] = create_data(test_samples);
        y_test = (class_labels_test==1)*(0) + (class_labels_test==2)*(1); % If class 1, y=0. If class 2, y=1.
        regression_targets_test = NaN;  % Wont be used for classification.
        X_test = cat(2, ones(test_samples,1), X_test); % concat 1 for bias

        [X_train, y_train, X_val, y_val] = divide_data(X, y, n_samples(i));
        
        [theta_opt, mean_ll] = logistic_regression_gd(X_train, y_train, lambda, n_iters);
        train_classification_errors(j,i) = classif_error(y_train, log_regr(X_train, theta_opt));
        val_classification_errors(j,i) = classif_error(y_val, log_regr(X_val, theta_opt));
        test_classification_errors(j,i) = classif_error(y_test, log_regr(X_test, theta_opt));
    end
end

means_train = mean(train_classification_errors, 1);
stds_train = std(train_classification_errors, 1);
means_and_stds_train = cat(1, means_train, stds_train);
fprintf("Experiment Train Error Means:\n");
disp(means_and_stds_train(1,:));
fprintf("Experiment Train Error Standard Deviations:\n");
disp(means_and_stds_train(2,:));

means_val = mean(val_classification_errors, 1);
stds_val = std(val_classification_errors, 1);
means_and_stds_val = cat(1, means_val, stds_val);
fprintf("Experiment Validation Error Means:\n");
disp(means_and_stds_val(1,:));
fprintf("Experiment Validation Error Standard Deviations:\n");
disp(means_and_stds_val(2,:));

means_test = mean(test_classification_errors, 1);
stds_test = std(test_classification_errors, 1);
means_and_stds_test = cat(1, means_test, stds_test);
fprintf("Experiment Test Error Means:\n");
disp(means_and_stds_test(1,:));
fprintf("Experiment Test Error Standard Deviations:\n");
disp(means_and_stds_test(2,:));

% ........


function [theta_opt, mean_loglosses] = logistic_regression_gd(X_train, y_train, learning_rate, iters_total)
    n_features = size(X_train, 2);
    theta_curr = zeros(n_features, 1);
    mean_loglosses = zeros(1, iters_total);
    for i = 1:iters_total
        % predicted y
        y_pred = log_regr(X_train, theta_curr);
        % grad of log-loss
        grad = X_train' * (y_pred - y_train);
        % update theta
        theta_curr = theta_curr - learning_rate * 1 / length(X_train) * grad;
        mean_loglosses(1,i) = mean_logloss(X_train, y_train, theta_curr);
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
