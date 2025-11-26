

function y = ReverseBuffer(x,Window,Res)

n = size(x,2);

i = 1 : n;
i = buffer(i,Window,Res);

y = zeros(1,n);

for j = 1 : n

    indices = i==j;

    y(j) = mean(x(indices));

end

end





