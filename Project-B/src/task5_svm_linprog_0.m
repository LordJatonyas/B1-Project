clear all;
close all;
clc;


% Note: Remember to Change class labels {1,2} to values for y appropriate for SVM => y={-1,+1}

rng(12345) % Random number generator seed. Comment it out to use random seed.

% Create training data
n_samples = 1000;
[X_train, regression_targets_train, class_labels_train] = create_data(n_samples);
y_train = (class_labels_train==1)*(-1) + (class_labels_train==2)*(1); % If class 1, y=-1. If class 2, y=+1
regression_targets_train = NaN;  % Wont be used for classification.
X_train = cat(2, ones(n_samples,1), X_train); % concat 1 for bias

test_samples = 20000;
[X_test, regression_targets_test, class_labels_test] = create_data(test_samples);
y_test = (class_labels_test==1)*(-1) + (class_labels_test==2)*(1);
regression_targets_test = NaN;
X_test = cat(2, ones(test_samples, 1), X_test);
%...

% Optimize - Support Vector Machine - Linear Programming
theta_opt = train_SVM_linear_progr(X_train, y_train);
train_scores = svm(X_train, theta_opt);
train_err = classif_error(y_train, train_scores);
fprintf("Training Error:%.4f\n", train_err);
test_scores = svm(X_test, theta_opt);
test_err = classif_error(y_test, test_scores);
fprintf("Test Error:%.4f\n", test_err);

% Plotting the decision boundaries for SVM, SGD, and GD
x = -10:10:0.1;
plot(x * -0.3516 + 1.7635, "red");
title("Decision Boundaries");
xlabel("x^( ^1 ^)");
ylabel("x^( ^2 ^)");
hold on;
plot(x * -0.3033 + 1.7453, "blue");
plot(x * -0.3099 + 1.7614, "green");
legend("SVM", "SGD LR", "GD LR");
hold off;
saveas(gcf, "../report/task5-decision_boundaries.png");
%...


function theta_opt = train_SVM_linear_progr(X, y)
    f = [ones(length(y), 1); 0; 0; 0];
    b = -ones(length(y), 1);
    A = -eye(length(y));
    for i = 1:length(y)
        if y(i, 1) == 1
            X(i, :) = -X(i, :);
        end
    end
    A = [A X];
    Aeq = [];
    beq = [];
    lb = [zeros(length(y), 1); -inf; -inf; -inf];
    ub = [inf(length(lb), 1)];

    theta_opt = linprog(f, A, b, Aeq, beq, lb, ub);
    theta_opt = theta_opt(end-2:end, 1);
end

function class_score = svm(X, theta)
    class_score = X * theta;
    class_score = (class_score<0)*(-1) + (class_score>0)*(1);
end

function err_perc = classif_error(y_true, y_pred)
    wrong_count = sum(y_true ~= round(y_pred));
    err_perc = wrong_count / length(y_pred);
end



