close all;

% Initialize transition probabilities with 0.1
T = 0.1 * ones(10, 10);

% Update the transition probabilities
for i = 1:10
    T(i, mod(i, 10) + 1) = 0.5;  % Increase probability to transition to the next digit
    T(i, :) = T(i, :) / sum(T(i, :));  % Normalize to sum to 1
end

% Display the transition probabilities matrix
disp(T);
figure
imagesc(T)
colorbar
axis square
h = gca;
h.XTick = 0:9;
h.YTick = 0:9;
title("Transition Matrix Heatmap")
figure
stateNames=["0" "1" "2" "3" "4" "5" "6" "7" "8" "9"];
mc = dtmc(T,StateNames=stateNames);
graphplot(mc,ColorEdges=true)
