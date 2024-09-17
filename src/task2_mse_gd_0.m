clear all;
close all;
clc;


rng(12345) % Random number generator seed. Comment it out to use random seed.

% Create training/val data
n_samples = 1000;
[X_train_val, regression_targets_train_val, class_labels_train_val] = create_data(n_samples);
%... rest is for you to implement ...
%...
y_train_val = regression_targets_train_val;
class_labals_train_val = NaN; % Won't be used for regression
X_train_val = cat(2, ones(n_samples,1), X_train_val); % concat 1 for bias
[X_train, y_train, X_val, y_val] = divide_data(X_train_val, y_train_val, n_samples); % Train-Validation split 80-20
% 

% Optimize - Linear Regression - Gradient Descent
lambdas = [0.00001 0.0001 0.001 0.01 0.1 1];
mses = zeros(2, length(lambdas));

for i = 1:length(lambdas)
    lambda = lambdas(i);
    theta_opt = linear_regression_gd(X_train, y_train, lambda, 1000);
    mse_train = mean_squared_error(X_train, y_train, theta_opt);
    mse_val = mean_squared_error(X_val, y_val, theta_opt);
    mses(1,i) = mse_train;
    mses(2,i) = mse_val;
end
[mse_train_min, ind1] = min(mses(1,:), [], 'all', 'linear');
[mse_val_min, ind2] = min(mses(2,:), [], "all", "linear");
theta_opt = linear_regression_gd(X_train, y_train, lambdas(ind2), 1000);

fprintf("Training MSEs:\n");
disp(mses(1,:));
fprintf("Validation MSEs:\n");
disp(mses(2,:));
fprintf("Optimal Learning Rate: %.4f\n\n", lambdas(ind2));
fprintf("Bias: %.4f\n", theta_opt(1));
fprintf("Parameter 1: %.4f\n", theta_opt(2));
fprintf("Parameter 2: %.4f\n\n", theta_opt(3));

% Create test data
test_samples = 20000;
[X_test, regression_targets_test, class_labels_test] = create_data(test_samples);
y_test = regression_targets_test;
class_labels_test = NaN; % Won't be used for regression
X_test = cat(2, ones(test_samples,1), X_test); % concat 1 for bias

% Check Test MSE
mse_test = mean_squared_error(X_test, y_test, theta_opt);
fprintf("Test MSE (with Learning Rate = 0.1): %.4f\n\n", mse_test);

% Rerun experiment with n_iters = 10000
mses = zeros(2, length(lambdas));

for i = 1:length(lambdas)
    lambda = lambdas(i);
    theta_opt = linear_regression_gd(X_train, y_train, lambda, 10000);
    mse_train = mean_squared_error(X_train, y_train, theta_opt);
    mse_val = mean_squared_error(X_val, y_val, theta_opt);
    mses(1,i) = mse_train;
    mses(2,i) = mse_val;
end
[mse_train_min, ind1] = min(mses(1,:), [], 'all', 'linear');
[mse_val_min, ind2] = min(mses(2,:), [], "all", "linear");
theta_opt = linear_regression_gd(X_train, y_train, lambdas(ind2), 10000);

fprintf("Training MSEs (n_iters = 10000):\n");
disp(mses(1,:));
fprintf("Validation MSEs (n_iters = 10000):\n");
disp(mses(2,:));
fprintf("Optimal Learning Rate: %.4f\n\n", lambdas(ind2));

% Running Task 1 with the same 800 training data
theta_opt_1 = mse_regression_closed_form(X_train, y_train);
fprintf("Bias (Task 1): %.4f\n", theta_opt_1(1));
fprintf("Parameter 1 (Task 1): %.4f\n", theta_opt_1(2));
fprintf("Parameter 2 (Task 1): %.4f\n\n", theta_opt_1(3));
%...

% Training-Validation Split
function [X_train, y_train, X_val, y_val] = divide_data(X, y, n)
    indices = randperm(n);
    test_cap = n * 0.8;
    X_train = X(indices(1:test_cap), :);
    y_train = y(indices(1:test_cap), 1);
    X_val = X(indices(test_cap+1:n), :);
    y_val = y(indices(test_cap+1:n), 1);
end

% GD initialises theta with all 0
% Compute the value of the cost function for current parameters
% Compute gradient of cost function
% 
function theta_opt = linear_regression_gd(X_train, y_train, learning_rate, iters_total)
    %....
    m = length(X_train);
    n_features = size(X_train, 2);
    % Initialize theta
    theta_curr = zeros(n_features,1);  % Current theta

    for i = 1:iters_total
        % Compute gradients
        % ...
        grad = 2/m * X_train' * (X_train * theta_curr - y_train);

        % Update theta
        % ...
        theta_curr = theta_curr - learning_rate * grad;
    end
    theta_opt = theta_curr;
end


function mse = mean_squared_error(X, y, theta)
    mse = 1 / length(X) * (X * theta - y)' * (X * theta - y);
end

function theta_opt = mse_regression_closed_form(X, y)
    theta_opt = (X' * X)\X' * y;
end

