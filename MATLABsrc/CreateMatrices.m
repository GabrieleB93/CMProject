function CreateMatrices(type)


% A = Matrix 1000x50, x E [-range,range], density = 1
% B = Matrix 1000x50, x E [-range,range], density = 0.5
% C = Matrix 1000x5, x E [-range,range], density = 0.25
% D = Matrix 1000x5, x E [-range,range], density = 0.01
% E = Matrix 1000x1000, 0<x<1, density = 1, ill conditioned

    num = 10;
    Homedirectory = ('Matrices');
    if (~exist(Homedirectory, 'dir')); mkdir(Homedirectory); end%if
    directory = strcat(Homedirectory,'/','Matrix',type);
    if (~exist(directory, 'dir')); mkdir(directory); end%if
    m = 1000;
    n = 50;
    n1 = 5;
    for i = 1:num

        range = randi(100);

        switch type
            case 'A'
                r = -range + (range+range)*randn(m,n);
            case 'B'
                r = sparseR(m,n,0.5,range);
            case 'C'
                r = sparseR(m,n1,0.25,range);
            case 'D'
                r = sparseR(m,n1,0.01,range);
            case 'G'
                r = golub(1000);
                disp(cond(r));
            otherwise
                disp('Inserire lettera valida');
                return
        end

        filename = strcat(directory,'/','matrix',type,num2str(i),'.txt');
        dlmwrite(filename,r,'delimiter','\t','precision',3)

    end

    function [r]  = sparseR(m,n,density,range)
        idx = randperm(m*n,round(density*m*n)); % Find random indices
        R = zeros(m,n);
        R(idx) = -range + (range+range)*rand(1,numel(idx));% Create random matrix
        r = full(sparse(R));