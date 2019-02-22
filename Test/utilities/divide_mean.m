function y = divide_mean(x)
x2col = x(:);
x_mean = mean(x2col);
y  = x./x_mean;

