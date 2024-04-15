function plotDecisionBoundary(theta, X, y, degree)
    plotData(X(:,2:3), y);
    hold on

    if size(X, 2) <= 3
        plot_x = [min(X(:,2)) - 2,  max(X(:,2)) + 2];

        plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

        plot(plot_x, plot_y)

        axis([min(X(:,2))-2, max(X(:,2))+2, ...
            min(X(:,3))-2, max(X(:,3))+2]);
    else
        % Задается диапазон сетки
        u = linspace(-1, 1.5, 50);
        v = linspace(-1, 1.5, 50);

        z = zeros(length(u), length(v));
        % Оценка z = theta*x на сетке
        for i = 1:length(u)
            for j = 1:length(v)
                z(i,j) = mapFeature(u(i), v(j), degree)*theta;
            end
        end
        z = z';

        contour(u, v, z, [0, 0], 'LineWidth', 2)
    end
    hold off
end
