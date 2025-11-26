clear; clc;

%% Parameters and Initial Conditions

t_start = 0;
t_end = 100;
dt = 1e-2;

t = t_start:dt:t_end;

a = 1.1;
b = 0.4;
c = 0.1;
d = 0.4;

x = zeros(size(t));
y = zeros(size(t));

x(1) = 7;
y(1) = 2;

%%

K = 100;

t_num = 0;

for i = 2 : length(t)

    x0 = x(i-1);
    y0 = y(i-1); 

    dt_num = 0;

    for k = 1 : K

        dt_sim = dt/K;

        x1 = x0 + (a*x0 - b*x0*y0)*dt_sim;
        y1 = y0 + (c*x0*y0 - d*y0)*dt_sim;

        x0 = x1;
        y0 = y1;

        dt_num = dt_num + dt_sim;

    end

    t_num(i) = t_num(i-1) + dt_num;
    x(i) = x1;
    y(i) = y1;

end


%%

figure(1)

subplot(2,2,1)
plot(t,x)
hold on

subplot(2,2,3)
plot(t,y)
hold on

subplot(2,2,[2 4])
plot(x,y)
hold on

LV.t = t;
LV.x = x;
LV.y = y;
LV.a = a;
LV.b = b;
LV.c = c;
LV.d = d;
LV.dt = dt;