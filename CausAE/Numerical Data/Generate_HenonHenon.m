%% 

clear; clc;

%%

N = 2e4;

%%

x1 = zeros(N,1); x2 = x1;
y1 = zeros(N,1); y2 = y1;

x1(1) = 0.7; x2(1) = 0;
y1(1) = 0.91; y2(1) = 0.7;

C = 0.64;

%%

for i = 2 : N

    x1(i) = 1.4 - x1(i-1).^2 + 0.3.*x2(i-1);
    x2(i) = x1(i-1);

    y1(i) = 1.4 - (C*x1(i-1)*y1(i-1) + (1-C).*y1(i-1).^2) + 0.3.*y2(i-1);
    y2(i) = y1(i-1);
    
end

%%

t = 1:1:N;

figure(1)
clf
subplot(2,2,1)
plot(t,x1)

subplot(2,2,3)
plot(t,y1)

subplot(2,2,2)
plot(x1,x2,'.')

subplot(2,2,4)
plot(y1,y2,'.')

% x = x';
% y = y';

clear i

save("HenonHenon_C0"+floor(C*100)+".mat")