%% Copyright (c) 2020,  Xiaowei Gu and Plamen P. Angelov

%% All rights reserved. Please read the "license.txt" for license terms.

%% This code is the self-adaptive synthetic over-sampling (SASYNO)approach described in:
%==========================================================================================================
%% X. Gu, P. Angelov, E Soares "A self-adaptive synthetic over-sampling technique for imbalanced classification," 
%% International Journal of Intelligent Systems, DOI: 10.1002/int.22230, 2020.
%==========================================================================================================
%% Please cite the paper above if this code helps.

%% For any queries about the code, please contact Dr. Xiaowei Gu, Prof. Plamen Angelov and Mr. Eduardo Soares
%% {x.gu3,p.angelov,e.almeidasoares}@lancaster.ac.uk

%% Programmed by Xiaowei Gu
function [augdata,auglabel]=SASYNO(data,label)
W=length(data(1,:));
label1=unique(label);
data1={};
for ii=1:1:length(label1)
    seq=find(label==label1(ii));
    L1(ii)=length(seq);
    data1{ii}=data(seq,:);
end
L2=max(L1);
for ii=find(L1~=L2)
    %%
    averdist2=zeros(1,W);
    for jj=1:1:W
        dist1=pdist(data1{ii}(:,jj));
        averdist2(jj)=mean(dist1(dist1<=mean(dist1)));
    end
    Seq=ones(L1(ii));
    dist1=pdist(data1{ii});
    dist11=squareform(dist1);
    averdist=mean(dist1(dist1<=mean(dist1)));
    Seq(dist11>averdist)=0;
    Seq=Seq-eye(L1(ii));
    [a,b]=find(Seq==1);
    if L1(ii)<L2
        L3=L2-L1(ii);
    end
    seq2=ceil(rand(1,L3)*length(a));
    seq3=rand(L3,W);seq4=1-seq3;
    seq5=repmat(averdist2./2,L3,1).*randn(L3,W);
    seq6=repmat(averdist2./2,L3,1).*randn(L3,W);
    data1{ii}=[data1{ii};((data1{ii}(a(seq2),:)+seq6).*seq3+(data1{ii}(b(seq2),:)+seq5).*seq4)];
end
augdata=[];
auglabel=[];
for ii=1:1:length(label1)
    augdata=[augdata;data1{ii}];
    auglabel=[auglabel;ones(length(data1{ii}(:,1)),1)*ii];
end
end