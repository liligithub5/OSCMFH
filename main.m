
clc;clear 
load mirflickr25k.mat
%% Parameter setting
run = 5;
Bits = [16,32,64,128];
%% Preprocessing data 
numbatch = 2000;
[streamdata,streamdata_non,nstream,L_tr,I_tr,T_tr,I_tr_non,T_tr_non,I_te_non,T_te_non] = predata_stream(I_tr,T_tr,L_tr,I_te,T_te,numbatch);
for i = 1:length(Bits)
    for j=1:run

       %% --------------------OURS----------------------------%%
        [B_I,B_T,tB_I,tB_T] = main_OSCMFH(streamdata, I_te, T_te, Bits(i));
        Dhamm = hammingDist(tB_I, B_T)';    
        [~, HammingRank]=sort(Dhamm,1);
        mapIT = map_rank(L_tr,L_te,HammingRank); 
        Dhamm = hammingDist(tB_T, B_I)';    
        [~, HammingRank]=sort(Dhamm,1);
        mapTI = map_rank(L_tr,L_te,HammingRank); 
        map(j, 1) = mapIT(100);
        map(j, 2) = mapTI(100);
        
    end
fprintf('\nbits = %d\n', Bits(i));
fprintf('average map over %d runs for ImageQueryOnTextDB: %.4f, chunks_size:%d\n', run, mean(map( : , 1)),numbatch);
fprintf('average map over %d runs for TextQueryOnImageDB: %.4f, chunks_size:%d\n', run, mean(map( : , 2)),numbatch);
end

