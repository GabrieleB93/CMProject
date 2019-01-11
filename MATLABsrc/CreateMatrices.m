function CreateMatrices(type)


% A = Matrix 1000x50, x > 1, density = 1
% B = Matrix 1000x5, x > 1, density = 1
% C = Matrix 1000x50, 0 < x < 1, density = 1
% D = Matrix 1000x50, x > 1, density = 0.5
% E = Matrix 1000x5, x > 1, density = 0.5
% F = Matrix 1000x50, x > 1, density = 0.25
% G = Matrix 1000x5, x > 1, density = 0.25
% H = Matrix 1000x5, 0 < x < 1, density = random


    n = 10;      
    directory = strcat('Matrix',type);
    if (~exist(directory, 'dir')); mkdir(directory); end%if
    for i = 1:n
        
        range = randi(100);
        
        switch type
            case 'A'
                r = -range + (range+range)*randn(1000,50);
            case 'B'
                r = -range + (range+range)*randn(1000,5);
            case 'C'
                r = full(sprand(1000,50,1));
            case 'D'
                r = full(sprand(1000,50,0.5,[-range,range]));
            case 'E'
                r = full(sprand(1000,5,0.5,[-range,range]));
            case 'F'
                r = full(sprand(1000,50,0.25,[-range,range]));
            case 'G'
                r = full(sprand(1000,5,0.25,[-range,range]));
            case 'H'
                r = full(sprand(1000,5,rand()));
            otherwise
                disp('Inserire lettera valida');
                return
        end       
        
        filename = strcat(directory,'/','matrix',type,num2str(i),'.txt');
        dlmwrite(filename,r,'delimiter','\t','precision',3)
    end
    