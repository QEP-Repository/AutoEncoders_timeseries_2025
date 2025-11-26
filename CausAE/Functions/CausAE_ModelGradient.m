
function [gradients, Loss_MSE, Loss_Code] = CausAE_ModelGradient(dZ_in,dZ_out,parameters,CausAE,alpha,sigma)

[dZp_out,Code] = CausAE_Network_1(dZ_in,[],1,parameters,CausAE,sigma);

Loss_MSE = mean((dZp_out-dZ_out).^2,'all');

Loss_Code = mean((mean(Code,1)).^2);

Loss = Loss_MSE + alpha.*Loss_Code;

gradients = dlgradient(Loss,parameters);

end


