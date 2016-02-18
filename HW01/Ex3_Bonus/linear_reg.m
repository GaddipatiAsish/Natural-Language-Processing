% keysFile = fopen('keys.txt');
% line = fgetl(keysFile);
% x = str2num(line);
% 
% valuesFile = fopen('values.txt');
% line = fgetl(valuesFile);
% y = str2num(line);
% 
% save('linear_reg.mat', 'x','y');
X = ones(size(x, 2), 2);
X(:,2) = x';
Y = y';

% cut
cut = 1;
X = X(cut:end-cut,:);
Y = Y(cut:end-cut,:);

P = X\Y;

plot(x, y, 'r.');
refline(P(2), P(1));