function [Y, Y1, Y2, R, U1, U2, P1, P2, obj] = solveOSCMFH( X1, X2, L, lambda, belta, gamma, bits, maxIter )

%% random initialization
%col = size(X1,2);  %n:�����ݼ��������������ӵĸ���
col = size(L,1);
Ibits = bits/4;  %Ks�� �ض�Ǳ������ռ��ά��
Sbits = 3*bits/4;   %Ku��ͳһǱ������ռ��ά��
%Y = rand(Sbits, col);   %Vu:ͳһ��ʾ����
%Y1 = rand(Ibits, col);  %Vx:ͼ������ʾ����
%Y2 = rand(Ibits, col);  %Vy:�ı������ʾ����
Z = rand(Sbits, col);
Z1 = rand(Ibits, col);
Z2 = rand(Ibits, col);

R = rand(Ibits, Ibits);  %R:��ؾ���
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
    % update R����ؾ���
    R = Y2 * Y1' / (Y1 * Y1' + (gamma/belta) * eye(Ibits));  %eye(N):����N*N�ĵ�λ����
    
	% update U1�����Ͼ���ֽ��е�ͼ��Ǳ����������, U2�����Ͼ���ֽ��е��ı�Ǳ����������,
    %  P1���������ֽ��е�ͼ��Ǳ����������, P2���������ֽ��е��ı�Ǳ����������
    U1 = X1 * Y' / (Y * Y' + (gamma/lambda) * eye(Sbits));  
    U2 = X2 * Y' / (Y * Y' + (gamma/(1-lambda)) * eye(Sbits));
    P1 = X1 * Y1' / (Y1 * Y1' + (gamma/lambda) * eye(Ibits));
    P2 = X2 * Y2' / (Y2 * Y2' + (gamma/(1-lambda)) * eye(Ibits));
    
	% update Y ��Vu:ͳһ��ʾ����   
   % Y = (lambda * U1' * U1 + (1- lambda) * U2' * U2 + gamma * eye(Sbits)) \ (lambda * U1' * X1 + (1 - lambda) * U2' * X2);
    Z = pinv(lambda * U1' * U1 + (1- lambda) * U2' * U2 + gamma * eye(Sbits)) * (lambda * U1' * X1 * L' + (1 - lambda) * U2' * X2 * L') / A;
    
    % update Y1��Vx:ͼ������ʾ���� and Y2��Vy:�ı������ʾ����
   % Y1 = (lambda * P1' * P1 + belta * R'* R + gamma * eye(Ibits)) \ (lambda * P1' * X1 + belta * R'* Y2);
   % Y2 = ((1-lambda) * P2' * P2 + belta * eye(Ibits) + gamma * eye(Ibits)) \ ((1 - lambda) * P2' * X2  + belta * R * Y1);
    Z1 = pinv(lambda * P1' * P1 + belta * R'* R + gamma * eye(Ibits)) * (lambda * P1' * X1 * L' + belta * R'* Y2 * L') / A;
    Z2 = pinv((1-lambda) * P2' * P2 + (belta + gamma) * eye(Ibits)) * ((1 - lambda) * P2' * X2 * L' + belta * R * Y1 * L') / A;     
    % compute objective function
    % norm������n=norm(A,p) norm�����ɼ��㼸�ֲ�ͬ���͵ľ�����,����p�Ĳ�ͬ�ɵõ���ͬ�ķ���
      %p  ����ֵ 
      %1  ����A�����һ�к� 
      %2  ����A���������ֵ
      %inf  ����A�����һ�к� 
      %��fro��  A��A���Ļ��ĶԽ��ߺ͵�ƽ��������sqrt(sum(diag(A'*A)))
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

