clear all;
close all;
clc;

% Note: Remember to change class labels {1,2} to values for y appropriate for SVM => y={-1,+1}

rng(12345) % Random number generator seed. Comment it out to use random seed.

% Create training/val data
n_samples = 1000;
[X_train_val, regression_targets_train_val, class_labels_train_val] = create_data(n_samples);
y_train_val = (class_labels_train_val==1)*(-1) + (class_labels_train_val==2)*(1); % If class 1, y=-1. If class 2, y=+1
regression_targets_train_val = NaN;  % Wont be used for classification.
% concat 1 for bias
X_train_val = cat(2, ones(n_samples,1), X_train_val);

%...
[X_train, y_train, X_val, y_val] = divide_data(X_train_val, y_train_val, n_samples);

% Create test data
test_samples = 20000;
[X_test, regression_targets_test, class_labels_test] = create_data(test_samples);
y_test = (class_labels_test==1)*(-1) + (class_labels_test==2)*(1); % If class 1, y=-1. If class 2, y=+1
regression_targets_test = NaN;  % Wont be used for classification.
% concat 1 for bias
X_test = cat(2, ones(test_samples,1), X_test);

% Optimize - Support Vector Machine - Gradient Descent with Hinge Loss
lambdas = [0.00001 0.0001 0.001 0.01 0.1 1 10];
n_iters = 10000;
errors = zeros(3, 6);
Js = zeros(length(lambdas), n_iters);

for i = 1:length(lambdas)
    lambda = lambdas(i);
    [theta_opt, ave_losses] = train_SVM_hingeloss_gd(X_train, y_train, lambda, n_iters);
    train_scores = svm(X_train, theta_opt);
    errors(1, i) = classif_error(y_train, train_scores);
    val_scores = svm(X_val, theta_opt);
    errors(2, i) = classif_error(y_val, val_scores);
    test_scores = svm(X_test, theta_opt);
    errors(3, i) = classif_error(y_test, test_scores);
    Js(i, :) = ave_losses;
    if lambdas(i) == 1
        theta = theta_opt;
    end
end

disp(errors);
disp(theta);

plot(Js(6, :));
title("Average Loss against Number of Iterations");
xlabel("Number of Iterations");
ylabel("Average Loss");
saveas(gcf, "../report/task6-average_loss.png");

% Plotting the decision boundaries for SVM, SGD, and GD
x = -10:10:0.1;
plot(x * -0.3516 + 1.7635, "red");
title("Decision Boundaries");
xlabel("x^( ^1 ^)");
ylabel("x^( ^2 ^)");
hold on;
plot(x * -0.3546 + 1.7619, "blue");
legend("Hinge Loss GD", "Linear Programming");
hold off;
saveas(gcf, "../report/task6-decision_boundaries.png");
%...


function [theta_opt, ave_losses] = train_SVM_hingeloss_gd(X_train, y_train, learning_rate, iters_total)
    % Initialize parameters
    n_features = size(X_train, 2);
    theta_curr = zeros(n_features,1);
    ave_losses = zeros(1, iters_total);
    for i = 1:iters_total
        grad_loss = zeros(n_features, 1);
        hinge_losses = hinge_loss_per_sample(X_train, y_train, theta_curr);
        for j = 1:length(hinge_losses)
            if hinge_losses(j) > 0
                grad_loss = grad_loss - y_train(j) * X_train(j, :)';
            end
        end
        ave_losses(i) = 1 / length(hinge_losses) * sum(hinge_losses);
        theta_curr = theta_curr - learning_rate / length(y_train) * grad_loss;
    end
    theta_opt = theta_curr;
end


function loss_per_sample = hinge_loss_per_sample(X, y_true, theta)
    y = X * theta;
    for i = 1:length(y)
        y(i) = 1 - y_true(i) * y(i);
        y(i) = max([0 y(i)]);
    end
    loss_per_sample = y;
end

function class_score = svm(X, theta)
    class_score = X * theta;
    class_score = (class_score<0)*(-1) + (class_score>0)*(1);
end

function err_perc = classif_error(y_true, y_pred)
    wrong_count = sum(y_true ~= round(y_pred));
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