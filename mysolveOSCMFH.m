function [L, U1, U2, P1, P2, E1, E2, E3, E4, C1, C2, C3, C4, F, W, HH, HX, HY, HX1, HY1, R, Vt] = mysolveOSCMFH(Itrain, Ttrain, L_tr, L, U1, U2, P1, P2, E1, E2, E3, E4, C1, C2, C3, C4, F, W, HH, HX, HY, HX1, HY1, R, lambda, mu, gamma, numiter)

A = L_tr * L_tr';
Z = pinv(lambda * U1' * U1 + (1- lambda) * U2' * U2 + gamma * eye(size(U1,2))) * (lambda * U1' * Itrain * L_tr' + (1 - lambda) * U2' * Ttrain * L_tr') / A;
Z1 = pinv(lambda * P1' * P1 + mu * R'* R + gamma * eye(size(P1,2))) * (lambda * P1' * Itrain * L_tr' + mu * R'* HY * L') / A;
Z2 = pinv((1-lambda) * P2' * P2 + (mu + gamma) * eye(size(P2,2))) * ((1 - lambda) * P2' * Ttrain * L_tr' + mu * R * HX * L') / A;  


Uold = [lambda*U1;(1-lambda)*U2];
P1old = lambda*P1;
P2old = (1-lambda)*P2;

threshold = 0.1;
lastF = 99999999;
%% Update Parameters
for i = 1:numiter
    H = Z * L_tr;
    HX1 = Z1 * L_tr;
    HY1 = Z2 * L_tr;
    % update U1 and U2    
    E1 = E1 + Itrain * H';
    C1 = C1 + H * H';
    
    E2 = E2 + Ttrain * H';
    C2 = C2 + H * H';
    
    U1 = E1 / C1;
    U2 = E2 / C2;
    
    % update P1 and P2
    E3 = E3 + Itrain * HX1';
    C3 = C3 + HX1 * HX1';
    
    E4 = E4 + Ttrain * HY1';
    C4 = C4 + HY1 * HY1';
    
    P1 = E3 / C3;
    P2 = E4 / C4;

    % update R
    F = F + HY1 * HX1';
    W = W + HX1 * HX1';
    R = F / W;
    
    
    % update Z 、Z1 、Z2    
    Z = pinv(lambda * U1' * U1 + (1- lambda) * U2' * U2 + gamma * eye(size(U1,2))) * (lambda * U1' * Itrain * L_tr' + (1 - lambda) * U2' * Ttrain * L_tr') / A;
    Z1 = pinv(lambda * P1' * P1 + mu * R'* R + gamma * eye(size(P1,2))) * (lambda * P1' * Itrain * L_tr' + mu * R'* HY1 * L_tr') / A;
    Z2 = pinv((1-lambda) * P2' * P2 + (mu + gamma) * eye(size(P2,2))) * ((1 - lambda) * P2' * Ttrain * L_tr' + mu * R * HX1 * L_tr') / A; 
    
    
    % update PI and PT
    
%    F1 = F1 + V*Itrain';   
%    W1 = W1 + Itrain*Itrain';

%    F2 = F2 + V*Ttrain';
%   W2 = W2 + Ttrain*Ttrain';

%    PI = F1 / W1;
%    PT = F2 / W2;
    
    H = Z * L_tr;
    HX1 = Z1 * L_tr;
    HY1 = Z2 * L_tr;
    % compute objective function
    norm1 = lambda * (norm(Itrain - U1 * H, 'fro') + norm(Itrain - P1 * HX1, 'fro'));
    norm2 = (1 - lambda) * (norm(Ttrain - U2 * H, 'fro') + norm(Ttrain - P2 * HY1, 'fro'));
    norm3 = mu * norm(HY1 - R * HX1, 'fro');
    norm4 = gamma * (norm(U1, 'fro') + norm(U2, 'fro') + norm(P1, 'fro') + norm(P2, 'fro') + norm(H, 'fro') + norm(HX1, 'fro')  + norm(HY1, 'fro'));
    currentF= norm1 + norm2 + norm3 + norm4;
    if abs(lastF - currentF) < threshold
%        fprintf('algorithm converges...\n');
%        fprintf('umber of iterations: %d\n reconstruction error : %.4f\n\n',i, currentF);
        Vt = [H;HY1];
        % update HH
        Unew = [lambda*U1;(1-lambda)*U2];
        Znew = (Unew' * Unew + gamma * eye(size(U1,2)))\(Unew' * Uold * HH * L')/(L * L');  %上一轮的Z在本轮进行更新
        HH = [Znew * L,Z * L_tr];

        % update HX
        P1new = lambda*P1;
        Z1new = (P1new' * P1new + gamma * eye(size(P1,2)))\(P1new' * P1old * HX * L')/(L * L');  %上一轮的Z在本轮进行更新
        HX = [Z1new * L,Z1 * L_tr];

        % update HY
        P2new = (1-lambda)*P2;
        Z2new = (P2new' * P2new + gamma * eye(size(P2,2)))\(P2new' * P2old * HY * L')/(L * L');  %上一轮的Z在本轮进行更新
        HY = [Z2new * L,Z2 * L_tr];
        L = [L,L_tr];
        return
    end
    lastF = currentF;
end
    Vt = [H;HY1];
    % update HH
    Unew = [lambda*U1;(1-lambda)*U2];
    Znew = (Unew' * Unew + gamma * eye(size(U1,2)))\(Unew' * Uold * HH * L')/(L * L');  %上一轮的Z在本轮进行更新
    HH = [Znew * L,Z * L_tr];

    % update HX
    P1new = lambda*P1;
    Z1new = (P1new' * P1new + gamma * eye(size(P1,2)))\(P1new' * P1old * HX * L')/(L * L');  %上一轮的Z在本轮进行更新
    HX = [Z1new * L,Z1 * L_tr];

    % update HY
    P2new = (1-lambda)*P2;
    Z2new = (P2new' * P2new + gamma * eye(size(P2,2)))\(P2new' * P2old * HY * L')/(L * L');  %上一轮的Z在本轮进行更新
    HY = [Z2new * L,Z2 * L_tr];
    L = [L,L_tr];
     
end
