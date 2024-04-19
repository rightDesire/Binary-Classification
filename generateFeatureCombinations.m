function remaining_combinations = generateFeatureCombinations(num_features)
    total_combinations = 2^(num_features - 3);
    remaining_combinations = zeros(total_combinations, num_features);
    
    % Генерация комбинаций
    for i = 1:total_combinations
        binary_rep = dec2bin(i - 1, num_features - 3); % Преобразуем числа от 0 до total_combinations - 1 в двоичный формат
        binary_rep = ['1', '1', '1', binary_rep]; % Добавляем первые три единицы
        
        % Преобразование символов в числа
        remaining_combinations(i, :) = arrayfun(@(x) str2double(x), binary_rep);
    end
end
