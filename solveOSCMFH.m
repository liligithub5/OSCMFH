function [Y, Y1, Y2, R, U1, U2, P1, P2, obj] = solveOSCMFH( X1, X2, L, lambda, belta, gamma, bits, maxIter )

%% random initialization
%col = size(X1,2);  %n:在数据集中所包含的例子的个数
col = size(L,1);
Ibits = bits/4;  %Ks： 特定潜在语义空间的维数
Sbits = 3*bits/4;   %Ku：统一潜在语义空间的维数
%Y = rand(Sbits, col);   %Vu:统一表示矩阵
%Y1 = rand(Ibits, col);  %Vx:图像具体表示矩阵
%Y2 = rand(Ibits, col);  %Vy:文本具体表示矩阵
Z = rand(Sbits, col);
Z1 = rand(Ibits, col);
Z2 = rand(Ibits, col);

R = rand(Ibits, Ibits);  %R:相关矩阵
threshold = 0.1;
lastF = 99999999;
iter = 1;
% obj = zeros(maxIter, 1);
obj = [];
A = L * L';
%% compute iteratively
while (true)
    Y = Z * L;
    Y1 = Z1 * L;
    Y2 = Z2 * L;
    % update R（相关矩阵）
    R = Y2 * Y1' / (Y1 * Y1' + (gamma/belta) * eye(Ibits));  %eye(N):产生N*N的单位矩阵
    
	% update U1（联合矩阵分解中的图像潜在向量矩阵）, U2（联合矩阵分解中的文本潜在向量矩阵）,
    %  P1（个别矩阵分解中的图像潜在向量矩阵）, P2（个别矩阵分解中的文本潜在向量矩阵）
    U1 = X1 * Y' / (Y * Y' + (gamma/lambda) * eye(Sbits));  
    U2 = X2 * Y' / (Y * Y' + (gamma/(1-lambda)) * eye(Sbits));
    P1 = X1 * Y1' / (Y1 * Y1' + (gamma/lambda) * eye(Ibits));
    P2 = X2 * Y2' / (Y2 * Y2' + (gamma/(1-lambda)) * eye(Ibits));
    
	% update Y （Vu:统一表示矩阵）   
   % Y = (lambda * U1' * U1 + (1- lambda) * U2' * U2 + gamma * eye(Sbits)) \ (lambda * U1' * X1 + (1 - lambda) * U2' * X2);
    Z = pinv(lambda * U1' * U1 + (1- lambda) * U2' * U2 + gamma * eye(Sbits)) * (lambda * U1' * X1 * L' + (1 - lambda) * U2' * X2 * L') / A;
    
    % update Y1（Vx:图像具体表示矩阵） and Y2（Vy:文本具体表示矩阵）
   % Y1 = (lambda * P1' * P1 + belta * R'* R + gamma * eye(Ibits)) \ (lambda * P1' * X1 + belta * R'* Y2);
   % Y2 = ((1-lambda) * P2' * P2 + belta * eye(Ibits) + gamma * eye(Ibits)) \ ((1 - lambda) * P2' * X2  + belta * R * Y1);
    Z1 = pinv(lambda * P1' * P1 + belta * R'* R + gamma * eye(Ibits)) * (lambda * P1' * X1 * L' + belta * R'* Y2 * L') / A;
    Z2 = pinv((1-lambda) * P2' * P2 + (belta + gamma) * eye(Ibits)) * ((1 - lambda) * P2' * X2 * L' + belta * R * Y1 * L') / A;     
    % compute objective function
    % norm函数：n=norm(A,p) norm函数可计算几种不同类型的矩阵范数,根据p的不同可得到不同的范数
      %p  返回值 
      %1  返回A中最大一列和 
      %2  返回A的最大奇异值
      %inf  返回A中最大一行和 
      %‘fro’  A和A‘的积的对角线和的平方根，即sqrt(sum(diag(A'*A)))
    Y = Z * L;
    Y1 = Z1 * L;
    Y2 = Z2 * L;
    norm1 = lambda * norm(X1 - U1 * Y, 'fro') + (1 - lambda) * norm(X2 - U2 * Y, 'fro');
    norm2 = lambda * norm(X1 - P1 * Y1, 'fro') + (1 - lambda) * norm(X2 - P2 * Y2, 'fro');
    norm3 = belta * norm(Y2 - R * Y1, 'fro');
    norm4 = gamma * (norm(U1, 'fro') + norm(U2, 'fro') + norm(Y, 'fro') + norm(P1, 'fro') + norm(P2, 'fro') + norm(Y1, 'fro') + norm(Y2, 'fro') + norm(R, 'fro'));
    currentF= norm1 + norm2 + norm3 + norm4;
    obj = [obj,currentF];
%    fprintf('\nobj at iteration %d: %.4f\n reconstruction error : %.4f\n\n', iter, currentF);
    if abs(lastF - currentF) < threshold
%        fprintf('algorithm converges...\n');
%        fprintf('final obj: %.4f\n reconstruction error for collective matrix factorization: %.4f,\n reconstruction error for seperable matrix factorization: %.4f,\n reconstruction error for consistency: %.4f,\n regularization term: %.4f\n\n', currentF, norm1, norm2, norm3, norm4);
        return;
    end
    if iter>=maxIter
        return
    end
    iter = iter + 1;
    lastF = currentF;
end
end

