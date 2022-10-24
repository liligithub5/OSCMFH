function [B_Ir,B_Tr,B_Ie,B_Te,obj,traintime,testtime] = main_OSCMFH(streamdata, I_te, T_te, bits, lambda, mu, gamma, iter)

if ~exist('lambda','var')
    lambda = 0.1;
end
if ~exist('mu','var')
    mu = 0.01;
end
if ~exist('gamma','var')
    gamma = 0.01;
end
if ~exist('iter','var')
    iter = 40;
end

%% Training
traintime1 = cputime;
nstream = size(streamdata,2);
% Initialization
Itrain = streamdata{1,1};  Ttrain = streamdata{2,1};    L_tr = streamdata{3,1};
numdata = size(Itrain,1);
mean_I = mean(Itrain, 1);
mean_T = mean(Ttrain, 1);
Itrain = bsxfun(@minus, Itrain, mean_I);
Ttrain = bsxfun(@minus, Ttrain, mean_T);
mean_I = mean_I';
mean_T = mean_T'; 
%disp(['Batch:' num2str(1)  ' Total:' num2str(nstream)]);
[HH, HX, HY, R, U1, U2, P1, P2, obj] = solveOSCMFH(Itrain', Ttrain', L_tr', lambda, mu, gamma, bits, iter);

V = [HH;HY];  %V ：潜在语义表示矩阵
%P1，P2：线性投影矩阵
PI = V * Itrain / (Itrain' * Itrain + gamma * eye(size(Itrain,2)));
PT = V * Ttrain / (Ttrain' * Ttrain + gamma * eye(size(Ttrain,2)));

% Training:2--n
mFea1 = size(Itrain,2);
mFea2 = size(Ttrain,2);
E1 = Itrain' * HH';
E2 = Ttrain' * HH';
E3 = Itrain' * HX';
E4 = Ttrain' * HY';
C1 = HH * HH' + (gamma / lambda) * eye(size(HH,1));      
C2 = C1; 
C3 = HX * HX' + (gamma / lambda) * eye(size(P1,2)); 
C4 = HY * HY' + (gamma / lambda) * eye(size(P2,2)); 
F = HY * HX';
F1 = V*Itrain;
F2 = V*Ttrain;
W = HX *HX'+ (gamma / mu) * eye(size(HX,1));
W1 = Itrain'*Itrain + gamma * eye(mFea1);
W2 = Ttrain'*Ttrain + gamma * eye(mFea2);
HX1 = HX;
HY1 = HY;
L = L_tr';

for i = 2:nstream 
    Itrain = streamdata{1,i}';  Ttrain = streamdata{2,i}';  L_tr = streamdata{3,i}';
    numdata_tmp = size(Itrain,2);
    mean_Itmp = mean(Itrain, 2);
    mean_Ttmp = mean(Ttrain, 2);
    mean_I = (numdata*mean_I + numdata_tmp*mean_Itmp)/(numdata + numdata_tmp);
    mean_T = (numdata*mean_T + numdata_tmp*mean_Ttmp)/(numdata + numdata_tmp);
    Itrain = bsxfun(@minus, Itrain, mean_I);
    Ttrain = bsxfun(@minus, Ttrain, mean_T);
    numdata = numdata + numdata_tmp;
%    disp(['Batch:' num2str(i)  ' Total:' num2str(nstream)]);
    W1 = W1 + Itrain*Itrain';
    W2 = W2 + Ttrain*Ttrain';   
    [L, U1, U2, P1, P2, E1, E2, E3, E4, C1, C2, C3, C4, F, W, HH, HX, HY, HX1, HY1, R, Vt] = mysolveOSCMFH(Itrain, Ttrain, L_tr, L, U1, U2, P1, P2, E1, E2, E3, E4, C1, C2, C3, C4, F, W, HH, HX, HY, HX1, HY1, R, lambda, mu, gamma, iter);
      
    F1 = F1 + Vt*Itrain';  
    F2 = F2 + Vt*Ttrain';
    PI = F1 / W1;
    PT = F2 / W2;
end
V = [HH;HY];  
%% Calculate hash codes
Y_tr = sign((bsxfun(@minus, V , mean(V,2)))');
Y_tr(Y_tr<0) = 0;
B_Tr = compactbit(Y_tr);
B_Ir = B_Tr;
traintime2 = cputime;
traintime = traintime2 - traintime1;

testtime1 = cputime;
Yi_te = sign((bsxfun(@minus,PI * I_te' , mean(V,2)))');
Yt_te = sign((bsxfun(@minus,PT * T_te' , mean(V,2)))');
Yi_te(Yi_te<0) = 0;
Yt_te(Yt_te<0) = 0;
B_Te = compactbit(Yt_te);
B_Ie = compactbit(Yi_te);
testtime2 = cputime;
testtime = testtime2 - testtime1;