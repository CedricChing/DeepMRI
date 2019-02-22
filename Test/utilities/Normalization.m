function [im_input_12chs,im_label_1ch,im_input_1ch]=Normalization(Img,mask)
   
   mask_12chs= repmat(mask,[1 1 12]);
   
        % prepare label-data
        im_label_12chs = abs(Img);
        im_label_sos = sqrt(sum(im_label_12chs.^2,3));    
        im_label_1ch_temp = mapminmax(im_label_sos(:)',0,1);
        im_label_1ch = reshape(im_label_1ch_temp,size(Img,1),size(Img,2));
        
         % prepare input-data
         Kdata_12chs_complex = fft2c(Img);    
         im_input_12chs_complex = ifft2c(Kdata_12chs_complex.*mask_12chs);  
         im_input_12chs_abs = abs(im_input_12chs_complex);
            im_input_sos = sqrt(sum(im_input_12chs_abs.^2,3)); 
            im_input_1ch = im_input_sos./max(im_input_sos(:));
         im_input_12chs_nor = zeros(size(im_input_12chs_abs));
         for k = 1:12
             im_input_kch = im_input_12chs_abs(:,:,k);
             im_input_kch_temp = mapminmax(im_input_kch(:)',0,1);
             im_input_kch_nor = reshape(im_input_kch_temp,size(Img,1),size(Img,2));
             im_input_12chs_nor(:,:,k) = im_input_kch_nor;
         end 
        im_input_12chs = im_input_12chs_nor;   
       