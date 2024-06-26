function out = mapFeatureDegree(X1, X2, degree)
    out = ones(size(X1(:,1)));
    if (degree > 1)
        for i = 1:degree
            for j = 0:i
                out(:, end+1) = (X1.^(i-j)).*(X2.^j);
            end
        end
    else
        % Кастомный признак при degree = 0/1
        out(:, end+1) = X1;
        out(:, end+1) = X2;
    end
end
