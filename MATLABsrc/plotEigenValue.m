function plotEigenValue(A)
x = sym('x', 'real');
y = sym('y', 'real');
z = [x;y];
Q = A'*A;
f = (z'*Q*z)/(z'*z);
H = hessian(f, [x,y]);
ei = eig(H);
h = ei(1);
g = ei(2);
subplot(2,1,1);
fsurf(h);
subplot(2,1,2);
fsurf(g < 0);
end

