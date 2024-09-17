clear all;
close all;
clc;


rng(12345) % Random number generator seed. Comment it out to use random seed.

% Create training data
n_samples_train = 1000;
[X_train, regression_targets_train, class_labels_train] = create_data(n_samples_train);
y_train = regression_targets_train;  % Target for regression model - what to predict
class_labels_train = NaN;  % Wont be used for regression.
% concat 1 for bias
X_train = cat(2, ones(n_samples_train,1), X_train);

% Create testing data
n_samples_test = 20000;
[X_test, regression_targets_test, class_labels_test] = create_data(n_samples_test);
y_test = regression_targets_test;
class_labels_test = NaN;  % Wont be used for regression.
% concat 1 for bias
X_test = cat(2, ones(n_samples_test,1), X_test);

%... rest is up to you ...
%...
theta = mse_regression_closed_form(X_train, y_train);
fprintf("Bias: %.4f\n", theta(1));
fprintf("Parameter 1: %.4f\n", theta(2));
fprintf("Parameter 2: %.4f\n\n", theta(3));

mse_train = mean_squared_error(X_train, y_train, theta);
fprintf("Training MSE: %.4f\n", mse_train);

mse_test = mean_squared_error(X_test, y_test, theta);
fprintf("Test MSE: %.4f\n\n", mse_test);

% Create training data with 10 samples
n_samples_train = 10;
[X_train, regression_targets_train, class_labels_train] = create_data(n_samples_train);
y_train = regression_targets_train;  % Target for regression model - what to predict
class_labels_train = NaN;  % Wont be used for regression.
% concat 1 for bias
X_train = cat(2, ones(n_samples_train,1), X_train);

theta = mse_regression_closed_form(X_train, y_train);
fprintf("Bias (with 10 samples): %.4f\n", theta(1));
fprintf("Parameter 1 (with 10 samples): %.4f\n", theta(2));
fprintf("Parameter 2 (with 10 samples): %.4f\n\n", theta(3));

mse_train = mean_squared_error(X_train, y_train, theta);
fprintf("Training MSE (with 10 samples): %.4f\n", mse_train);

mse_test = mean_squared_error(X_test, y_test, theta);
fprintf("Test MSE (with 10 samples): %.4f\n\n", mse_test);


nsts = [4 10 20 100 1000 10000];
seeds = 1:20;
mses_train = zeros(20,6);
mses_test = zeros(20,6);

col_idx = 1;
for nst = nsts
    row_idx = 1;
    for s = seeds
        rng(s);
        [X_train, regression_targets_train, class_labels_train] = create_data(nst);
        y_train = regression_targets_train;  % Target for regression model - what to predict
        class_labels_train = NaN;  % Wont be used for regression.
        % concat 1 for bias
        X_train = cat(2, ones(nst,1), X_train);
        theta_opt = mse_regression_closed_form(X_train, y_train);

        mse_train = mean_squared_error(X_train, y_train, theta_opt);
        mses_train(row_idx, col_idx) = mse_train;

        % Create testing data
        n_samples_test = 20000;
        [X_test, regression_targets_test, class_labels_test] = create_data(n_samples_test);
        y_test = regression_targets_test;
        class_labels_test = NaN;  % Wont be used for regression.
        % concat 1 for bias
        X_test = cat(2, ones(n_samples_test,1), X_test);

        mse_test = mean_squared_error(X_test, y_test, theta_opt);

        mses_test(row_idx, col_idx) = mse_test;
        row_idx = row_idx + 1;
    end
    col_idx = col_idx + 1;
end

% Display Training Means and Standard Deviations
means_train = mean(mses_train, 1);
stds_train = std(mses_train, 1);
means_and_stds_train = cat(1, means_train, stds_train);

fprintf("Experimentation Training MSE Means:\n");
disp(means_and_stds_train(1,:));

fprintf("Experimentation Training MSE Standard Deviations:\n");
disp(means_and_stds_train(2,:));

% Display Test Means and Standard Deviations
means_test = mean(mses_test, 1);
stds_test = std(mses_test, 1);
means_and_stds_test = cat(1, means_test, stds_test);

fprintf("Experimentation Test MSE Means:\n");
disp(means_and_stds_test(1,:));

fprintf("Experimentation Test MSE Standard Deviations:\n");
disp(means_and_stds_test(2,:));


function mse = mean_squared_error(X, y, theta)
    % implement
    mse = 1 / length(X) * (X * theta - y)' * (X * theta - y);
end

function theta_opt = mse_regression_closed_form(X, y)
    % implement
    theta_opt = (X' * X)\X' * y;
end