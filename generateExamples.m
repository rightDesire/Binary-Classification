function [data_with_new_examples] = generateExamples(data, max_noise_level, num_examples_to_add)
    % Генерация новых примеров
    new_data = [];
    data_length = size(data, 1);

    for i = 1:num_examples_to_add
        % Выбор случайной строки из исходных данных
        idx = randi(data_length);
        x1_base = data(idx, 1);
        x2_base = data(idx, 2);
        label = data(idx, 3);

        % Рандомизация уровня шума для каждой координаты независимо
        noise_level_x1 = rand * max_noise_level;
        noise_level_x2 = rand * max_noise_level;

        % Создание новой точки с учетом шума или смещения
        x1 = x1_base + noise_level_x1;
        x2 = x2_base + noise_level_x2;

        % Добавление новой точки в список
        new_data = [new_data; x1, x2, label];
    end

    % Добавление новых примеров к исходным данным
    data_with_new_examples = [data; new_data];
end
