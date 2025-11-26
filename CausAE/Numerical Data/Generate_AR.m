%% 

clear; clc;

%%

N = 1e4;

%%

x = randn(N,1)/10;
y = randn(N,1)/10;

sigma = 0.1;

axx = 0.5;
ayy = 0.7;
axy = 0.2;

ayx = 0.4;

%%

for i = 2 : N

    x(i) = axx*x(i-1) + axy*y(i-1) + normrnd(0,sigma);
    y(i) = ayx*x(i-1) + ayy*y(i-1) + normrnd(0,sigma);

end

%%

t = 1:1:N;

figure(1)
clf
subplot(2,2,1)
plot(t,x)

subplot(2,2,3)
plot(t,y)

subplot(2,2,[2 4])
plot(x,y,'.')

x = x';
y = y';

clear i