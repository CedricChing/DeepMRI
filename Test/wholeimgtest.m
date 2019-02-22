% reconstruct sr image from corrupted image
% @author huitao cheng
close all; clear; clc;
%% initialize parameters
weights = 'demo.h5';
%% mask 1ch to 12chs
load test_data/1Duniform2.98_ac29.mat

mask = ifftshift(mask);
mask_12chs = repmat(mask,[1,1,12]);
imask_12chs = ~mask_12chs;
imask = ~mask;
%% load raw data-complex  and mask
addpath(genpath('test_data'));
addpath(genpath('utilities'));
addpath(genpath('result'));
folder_name = 'result/';
data_name = {'PD_1'};
data_num = numel(data_name);
for x = 1:data_num
    load(fullfile('test_data',data_name{x}));

    [hei,wid,cha]=size(Img);
    %% label data
    Img = Img ./ max(abs(Img(:)));
    im_label = AdaptiveCoilCombine(Img);

    %% input data
    Kdata_12chs_complex = fft2(Img);
    im_input = ifft2(Kdata_12chs_complex.*mask_12chs);

    im_input_1ch = AdaptiveCoilCombine(im_input);  % only for plot

    %% CVSRCNN reconstruction
    im_out_12chs = cvnnrecon(weights,im_input,mask_12chs);
    im_out_1ch = AdaptiveCoilCombine(im_out_12chs);
    im_out_sos = sos(im_out_12chs);
    %% variables for plot
    im_gnd_1ch = im_label;
    im_low_1ch = im_input_1ch; 
    im_rec_1ch = im_out_1ch;
    %% compute PSNR
    [psnr_low,ssim_low,rmse_low,error_low] = compute_psr_error_dm(abs(im_gnd_1ch),abs(im_low_1ch));
    [psnr_mrcnn,ssim_mrcnn,rmse_mrcnn,error_mrcnn]= compute_psr_error_dm(abs(im_gnd_1ch),abs(im_out_1ch));
    [psnr_sos,ssim_sos,rmse_sos,error_sos] = compute_psr_error_dm(abs(im_gnd_1ch),abs(im_out_sos));
    [psnr_rec,ssim_rec,rmse_rec,error_rec] = compute_psr_error_dm(abs(im_gnd_1ch),abs(im_rec_1ch));

    %% plot
    figure(1); imshow(abs(im_gnd_1ch),[]);
    set (gcf,'Position',[0,450,300,300]);
    set (gca,'position',[0 0 1 1]);
    saveas(figure(1), [folder_name, data_name{x}, '_', 'label', '.png'], 'png');
    figure(2); imshow(abs(im_low_1ch),[]); 
    set (gcf,'Position',[300,450,300,300]);
    set (gca,'position',[0 0 1 1]);
    saveas(figure(2), [folder_name, data_name{x}, '_', 'input', '.png'], 'png');
    figure(3); imshow(abs(im_out_sos),[]); 
    set (gcf,'Position',[600,450,355,295]);
    set (gca,'position',[0 0 1 1]);
    saveas(figure(3), [folder_name, data_name{x}, '_', 'output', '.png'], 'png');
    figure(4); imshow(abs(im_rec_1ch),[]); 
    set (gcf,'Position',[900,450,355,295]);
    set (gca,'position',[0 0 1 1]);
    saveas(figure(4), [folder_name, data_name{x}, '_', 'rec', '.png'], 'png');
    figure(5)
    imshow(error_rec,[0,0.1]), colormap jet; colorbar
    set (gcf,'Position',[1200,450,300,300]);
    set (gca,'position',[0.01 0 0.8 1]);
    saveas(figure(5), [folder_name, data_name{x}, '_', 'error', '.png'], 'png');
    save([folder_name, data_name{x},'_result.mat'],'im_input','im_label','Img','im_out_12chs','mask','psnr_rec','ssim_rec','rmse_rec','error_rec');
end
