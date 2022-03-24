addpath(genpath('./utils/'));
load(['./amazon_SURF_L10.mat']);
fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
Xs=fts';
Xs=double(zscore(Xs',1))';
Ys = labels;
load(['./caltech_SURF_L10.mat']);
fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
Xt=fts';
Xt=double(zscore(Xt',1))';
Yt = labels;

clear opt;opt.display=1;
% addpath(genpath('./2015-LRSR/'));
% LRSR(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2017-JGSA/'));
% JGSA(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2017-VDA/'));
% VDA(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2018-DICD/'));
% DICD(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2018-TLR/'));
% TLR(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2019-SPDA/'));
% SPDA(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2020-ATL/'));
% ATL(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2020-DAC/'));
% DAC(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2020-DGA/'));
% DGA(Xs,Ys,Xt,Yt,opt);



% addpath(genpath('./2020-DGSA/'));
% DGSA(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2020-DSL-DGDA/'));
% DSL_DGDA(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2020-McDA/'));
% McDA(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2020-WCS-RAR/'));
% WCS_RAR(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2021-CMFC/'));
% CMFC(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2021-DGB-DA/'));
% DGB_DA(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2021-LDA/'));
% LDA_DA(Xs,Ys,Xt,Yt,opt);

% addpath(genpath('./2021-PDALC/'));
% PDALC(Xs,Ys,Xt,Yt,opt);

