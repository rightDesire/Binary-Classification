%% Инициализация
clear ; close all; clc

%% Инициализация данных
num_features = 6;
lambda = 1;

% % Настройки генерации примеров
% num_examples_to_add = 1000;
% max_noise_level = 0.15;

data = load('ex2data2.txt');

% % Генерация новых примеров на основе исходных
% [data] = generateExamples(init_data, ...
%     max_noise_level, num_examples_to_add);

%% Загрузка данных
[mAll, mTraining, mTest, X_train, y_train, X_test, y_test] = ... 
    loadData(data);

%% Расчет КФО при моделях с разными степенями полиномов
polynomialDegreeAnalysis(X_train, y_train, X_test, y_test, lambda);

%% Варианты моделей до num_features признаков
featureCombinationsAnalysis(X_train, y_train, ...
    X_test, y_test, num_features, lambda);
