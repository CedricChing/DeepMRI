function param = GRAPPA_Experiment(Img,mask)
% addpath(strcat(pwd,'/utils'));
[~,~,coils] = size(Img);  % get sizes
DATA=fft2c(Img);
DATA = DATA.*repmat(mask,[1,1,coils]); % multiply with sampling matrix
temp_img=(ifft2c(DATA));
im_dc=sos(temp_img);
%%%%%%%%%%%%% prepare for grappa %%%%%%%%%%%%%%%%%
kSize = [5,5];  %calibrate a kernel
calibsize=getCalibSize(mask);
kCalib=crop(DATA,[calibsize,coils]);
lambda=0.01;
%%%%%%%%%%%%%%%%%% reconstruct %%%%%%%%%%%%%%%%%%%%%%:
tic
t0=cputime();
[res] = GRAPPA(DATA,kCalib, kSize, lambda);
t1 = cputime();
t = toc;
reconImg=sos(ifft2c(res));
MagRefImg=sos(Img);
%% normalization
% MagRefImg=mapminmax(MagRefImg(:)',0,1);
% reconImg=mapminmax(reconImg(:)',0,1);
% reconImg=reshape(reconImg,enum,pnum);
% MagRefImg=reshape(MagRefImg,enum,pnum);
% MagRefImg=MagRefImg./max(MagRefImg(:));
% reconImg=reconImg./max(reconImg(:));
%% compute quantitative index and save results
[param.psnr, param.ssim, param.rmse, param.error] = compute_psr_error_dm(MagRefImg, reconImg);
param.label = MagRefImg;
param.output = reconImg;
param.input_1ch = im_dc;
param.mask = mask;
param.kCalib = kCalib;
param.ksize = kSize;
param.lambda = lambda;
param.time_toc = t;
param.time_cputime = t1-t0;
param.time2 = t1-t0;
