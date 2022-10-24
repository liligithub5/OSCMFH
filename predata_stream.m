function [streamdata,streamdata_non,nstream,L_tr,I_tr,T_tr,I_tr_non,T_tr_non,I_te_non,T_te_non] = predata_stream(I_tr,T_tr,L_tr,I_te,T_te,numbatch)
%rand('seed',1);
anchors = 500;
anchor_idx = randsample(size(I_tr,1), anchors);
XAnchors = I_tr(anchor_idx,:);
anchor_idx = randsample(size(T_tr,1), anchors);
YAnchors = T_tr(anchor_idx,:);
[I_tr_non,I_te_non]=Kernel_Feature(I_tr,I_te,XAnchors);
[T_tr_non,T_te_non]=Kernel_Feature(T_tr,T_te,YAnchors);

[ndata,~] = size(I_tr);
Rdata = randperm(ndata);
I_tr = I_tr(Rdata,:);
T_tr = T_tr(Rdata,:);
L_tr = L_tr(Rdata,:);

I_tr_non = I_tr_non(Rdata,:);
T_tr_non = T_tr_non(Rdata,:);

nstream = ceil(ndata/numbatch);
streamdata = cell(3,nstream);
streamdata_non = cell(3,nstream);
for i = 1:nstream-1
    start = (i-1)*numbatch+1;
    endl = i*numbatch;
    streamdata{1,i} = I_tr(start:endl,:);
    streamdata{2,i} = T_tr(start:endl,:);
    streamdata{3,i} = L_tr(start:endl,:);
    
    streamdata_non{1,i} = I_tr_non(start:endl,:);
    streamdata_non{2,i} = T_tr_non(start:endl,:);
    streamdata_non{3,i} = L_tr(start:endl,:);
end
start = (nstream-1)*numbatch+1;
streamdata{1,nstream} = I_tr(start:end,:);
streamdata{2,nstream} = T_tr(start:end,:);
streamdata{3,nstream} = L_tr(start:end,:);

streamdata_non{1,nstream} = I_tr_non(start:end,:);
streamdata_non{2,nstream} = T_tr_non(start:end,:);
streamdata_non{3,nstream} = L_tr(start:end,:);

