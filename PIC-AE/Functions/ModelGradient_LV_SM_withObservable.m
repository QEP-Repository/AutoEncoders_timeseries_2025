
function [gradient,MSE,Physics,MSE_xy] = ModelGradient_LV_SM_withObservable(parameters,AE,dlX_0,dlX_1,alpha,Model,dlX_0b,dlx_b,dly_b)
    

    [X0_p,Code0] = AE_network_LV(dlX_0,[],1,parameters,AE,0);
    [X1_p,Code1] = AE_network_LV(dlX_1,[],1,parameters,AE,0);
    
    %% Reconstruction
    
    MSE0 = mean((X0_p-dlX_0).^2,'all');
    MSE1 = mean((X1_p-dlX_1).^2,'all');
    
    MSE = (MSE0+MSE1)/2;

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

    Physics = mean(fx.^2./std(x).^2 + fy.^2./std(y).^2);

    %%

    [X0_p,Code0] = AE_network_LV(dlX_0b,[],1,parameters,AE,0);

    MSE_xy = mean((Code0(1,:)-dlx_b).^2 + (Code0(2,:)-dly_b).^2);
        
    %% Total Loss and Graidents

    Loss = (MSE + alpha.*Physics + MSE_xy)./(1+alpha);

    gradient = dlgradient(Loss,parameters);

end



