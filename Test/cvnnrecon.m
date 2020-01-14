function im_hr = cvnnrecon(weightname, im_lr, mask)
tic
%% data process
t_real = real(im_lr);
t_imag = imag(im_lr);
k_lr = fft2(im_lr);
 %% block1
 for x=1:10
    [conv1_real,conv1_imag] = convolutionlayer(weightname,['/conv',num2str(x),'1'],t_real,t_imag,1);
    [conv2_real,conv2_imag] = convolutionlayer(weightname,['/conv',num2str(x),'2'],conv1_real,conv1_imag,1);
    [conv3_real,conv3_imag] = convolutionlayer(weightname,['/conv',num2str(x),'3'],conv2_real,conv2_imag,1);
    [conv4_real,conv4_imag] = convolutionlayer(weightname,['/conv',num2str(x),'4'],conv3_real,conv3_imag,1);
    [conv5_real,conv5_imag] = convolutionlayer(weightname,['/conv',num2str(x),'5'],conv4_real,conv4_imag,0);
    b_real = conv5_real + t_real;
    b_imag = conv5_imag + t_imag;
    %% dc
    dc = b_real + b_imag * 1i;
    k_dc = fft2(dc);
    t_out = ifft2(k_dc .* ~mask + k_lr .* mask);
    t_real = real(t_out);
    t_imag = imag(t_out);
 end
 
 im_hr_real = t_real ;
 im_hr_imag = t_imag ;
 im_hr = im_hr_real + im_hr_imag * 1i;
 
 toc