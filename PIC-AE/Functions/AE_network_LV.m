
function [X,Code,parameters] = AE_network_LV(X,Y,Predict,parameters,AE,Noise)

Layer_En = AE.Layer_En;
Layer_Dec = AE.Layer_Dec;

if Predict == 0

    CodeSize = AE.CodeSize;
    OutputSize = size(Y,1);
    
    %% Initialise net

    parameters = [];

    parameters.model.a = dlarray(1);
    parameters.model.b = dlarray(1);
    parameters.model.c = dlarray(1);
    parameters.model.d = dlarray(1);

    parameters.scale.Xmean = dlarray(max(X,[],2)+min(X,[],2))/2;
    parameters.scale.Xstd = dlarray(max(X,[],2)-min(X,[],2))/2;

    X = (X - parameters.scale.Xmean)./parameters.scale.Xstd; 

    for i = 1 : length(Layer_En)

        parameters.("en"+i).weights = dlarray(randn([Layer_En(i) size(X,1)])*sqrt(1/Layer_En(i)));
        parameters.("en"+i).bias = dlarray(zeros([Layer_En(i) 1]));

        X = fullyconnect(X,parameters.("en"+i).weights,parameters.("en"+i).bias);
        X = sigmoid(X).*X;

    end

    parameters.Code.weights = dlarray(randn([CodeSize size(X,1)]))/3;
    parameters.Code.bias = dlarray(zeros([CodeSize 1]));

    Code = fullyconnect(X,parameters.Code.weights,parameters.Code.bias);
    X = Code;
    
    for i = 1 : length(Layer_Dec)

        parameters.("dec"+i).weights = dlarray(randn([Layer_Dec(i) size(X,1)])*sqrt(2/Layer_Dec(i)));
        parameters.("dec"+i).bias = dlarray(zeros([Layer_Dec(i) 1]));

        X = fullyconnect(X,parameters.("dec"+i).weights,parameters.("dec"+i).bias);
        X = sigmoid(X).*X;

    end

    parameters.Output.weights = dlarray(randn([OutputSize size(X,1)]))/3;
    parameters.Output.bias = dlarray(zeros([OutputSize 1]));

    X = fullyconnect(X,parameters.Output.weights,parameters.Output.bias);

else

    X = (X - parameters.scale.Xmean)./parameters.scale.Xstd; 

    for i = 1 : length(Layer_En)

        X = fullyconnect(X,parameters.("en"+i).weights,parameters.("en"+i).bias);
        X = sigmoid(X).*X;

    end

    Code = fullyconnect(X,parameters.Code.weights,parameters.Code.bias);  
    
    Code = log(1+exp(Code));
    
    X = Code;
    
    for i = 1 : length(Layer_Dec)

        X = fullyconnect(X,parameters.("dec"+i).weights,parameters.("dec"+i).bias);
        X = sigmoid(X).*X;

    end

    X = fullyconnect(X,parameters.Output.weights,parameters.Output.bias);

end

end


