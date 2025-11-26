
function [gradient,MSE,Physics] = ModelGradient_LV_SM(parameters,AE,dlX_0,dlX_1,Noise,alpha,Model)
    

    [X0_p,Code0] = AE_network_LV_SM(dlX_0,[],1,parameters,AE,Noise);
    [X1_p,Code1] = AE_network_LV_SM(dlX_1,[],1,parameters,AE,Noise);
    
    %% Reconstruction
    
    MSE0 = mean((X0_p-dlX_0).^2,'all');
    MSE1 = mean((X1_p-dlX_1).^2,'all');
    
    MSE = MSE0+MSE1;

    %% Physics
    
    a = parameters.model.a;
    b = parameters.model.b;
    c = parameters.model.c;
    d = parameters.model.d;
    dt = Model.dt;

    x_1 = Code1(1,:); y_1 = Code1(2,:);
    x_0 = Code0(1,:); y_0 = Code0(2,:);

    dx = x_0-x_1; dy = y_0 - y_1;
    x = (x_0+x_1)/2; y = (y_0+y_1)/2;
    
    fx = dx - (a.*x - b.*x.*y).*dt; 
    fy = dy - (c.*x.*y - d.*y).*dt; 

    Physics = mean(fx.^2 + fy.^2);
        
    %% Total Loss and Graidents

    Loss = MSE + alpha.*Physics;

    gradient = dlgradient(Loss,parameters);

end



