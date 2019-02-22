function param = SPIRiT_Experiment(Img,mask)
[enum,pnum,coils] = size(Img);  % get sizes
DATA=fft2c(Img);
DATA = DATA.*repmat(mask,[1,1,coils]); % multiply with sampling matrix
temp_img=(ifft2c(DATA));
im_dc=sos(temp_img);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%% Prepare for SPIRiT
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%					     %	 			
kSize = [5,5];  % SPIRiT kernel size
nIterCG = 30; % number of iteration; 
CalibTyk = 0.01;  % Tykhonov regularization in the calibration
ReconTyk = 1e-5;  % Tykhovon regularization in the reconstruction (SPIRiT only)
[CalibSize, dcomp] = getCalibSize(mask);  % get size of calibration area from mask
% DATA = DATA.*repmat(mask,[1,1,coils]); % multiply with sampling matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Prepare for cgSPIRiT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('CG SPIRiT reconstruction ')
kCalib = crop(DATA,[CalibSize,coils]);
kernel = zeros([kSize,coils,coils]);
AtA = dat2AtA(kCalib,kSize);
for t=1:coils
    kernel(:,:,:,t) = calibrate(AtA,kSize,coils,t,CalibTyk);
end
GOP = SPIRiT(kernel, 'fft',[enum,pnum]);
tic
t0 = cputime();
res_cg = cgSPIRiT(DATA,GOP,nIterCG,ReconTyk, DATA);
t1 = cputime();
t = toc;
im_cgspirit = ifft2c(res_cg);
reconImg=sos(im_cgspirit);
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
param.time_toc = t;
param.time_cputime = t1-t0;
param.Calib=CalibTyk;
param.ReconTyk=ReconTyk;