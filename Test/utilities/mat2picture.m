function mat2picture(folder_name_string, error_scale_down, error_scale_up) % the input must be a character string

folder_name = folder_name_string;
folder_path = 'result\';
mat_num = length(dir(fullfile(['result\', folder_name], '*.mat')));

for i = 1:mat_num;
mat_file = dir(fullfile(['result\', folder_name], '*.mat'));
mat_name = mat_file(i).name;
load(['result\', folder_name, '\' mat_name])
figure(1)
imshow(param.label,[])
set (gcf,'Position',[0,450,300,300]);
set (gca,'position',[0 0 1 1]);
saveas(figure(1), [folder_path, folder_name, '\', mat_name, '_', 'label', '.png'], 'png');

figure(2)
imshow(param.input_1ch,[])
set (gcf,'Position',[300,450,300,300]);
set (gca,'position',[0 0 1 1]);
saveas(figure(2), [folder_path, folder_name, '\', mat_name, '_', 'input', '.png'], 'png');

figure(3)
imshow(param.output,[])
set (gcf,'Position',[600,450,300,300]);
set (gca,'position',[0 0 1 1]);
saveas(figure(3), [folder_path, folder_name, '\', mat_name, '_', 'output', '.png'], 'png');

figure(4)
imshow(param.mask,[])
set (gcf,'Position',[900,450,300,300]);
set (gca,'position',[0 0 1 1]);
saveas(figure(4), [folder_path, folder_name, '\', mat_name, '_', 'mask', '.png'], 'png');

figure(5)
imshow(param.error, [error_scale_down, error_scale_up]),colormap jet; colorbar
set (gcf,'Position',[1200,450,355,295]);
set (gca,'position',[0.01 0 0.8 1]);
% set (gcf,'Position',[650,450,300,300  ]);
% set (gca,'position',[0 0 1 1]);
saveas(figure(5), [folder_path, folder_name, '\', mat_name, '_', 'error', '.png'], 'png');

PSRT(i,:) = [param.psnr, param.ssim, param.rmse, param.time_toc, param.time_cputime];

end

mat_file_cell = struct2cell(mat_file)'; 
mat_name_all = mat_file_cell(:,1);  
tab = {'', 'PSNR', 'SSIM', 'RMSE', 'TIME'};
row_index = [mat_name_all, num2cell(PSRT)];
metrix = cat(1, tab, row_index);
xlswrite([folder_path, folder_name, '/', folder_name, '.xls'], metrix, 'Sheet1', 'A1');