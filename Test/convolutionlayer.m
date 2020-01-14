function [ conv_data_real, conv_data_imag ] = convolutionlayer( weightname,dbname,data_real,data_imag,activation )

    [hei,wid,~] = size(data_real);
    conv_kernel = h5read(weightname,[dbname,'/weights']);
    conv_kernel = permute(conv_kernel,[4,3,2,1]);
    [~,~,conv_channels,conv_filters] = size(conv_kernel);
    conv_filters = conv_filters / 2;
    conv_kernel_real = conv_kernel(:,:,:,1:conv_filters);
    conv_kernel_imag = conv_kernel(:,:,:,conv_filters+1:conv_filters*2);
    conv_bias = h5read(weightname,[dbname,'/biases']);
    conv_bias_real = conv_bias(1:conv_filters);
    conv_bias_imag = conv_bias(conv_filters+1:conv_filters*2);
    conv_data_real = zeros(hei,wid,conv_filters);
    conv_data_imag = zeros(hei,wid,conv_filters);
    for x = 1 : conv_filters
        for y = 1 : conv_channels
            Frr = imfilter(squeeze(data_real(:,:,y)),double(squeeze(conv_kernel_real(:,:,y,x))),'same','replicate');
            Fii = imfilter(squeeze(data_imag(:,:,y)),double(squeeze(conv_kernel_imag(:,:,y,x))),'same','replicate');
            Fri = imfilter(squeeze(data_real(:,:,y)),double(squeeze(conv_kernel_imag(:,:,y,x))),'same','replicate');
            Fir = imfilter(squeeze(data_imag(:,:,y)),double(squeeze(conv_kernel_real(:,:,y,x))),'same','replicate');
            conv_data_real(:,:,x) = conv_data_real(:,:,x) + Frr - Fii;
            conv_data_imag(:,:,x) = conv_data_imag(:,:,x) + Fri + Fir;
        end
        conv_data_real(:,:,x) = conv_data_real(:,:,x) + conv_bias_real(x);
        conv_data_imag(:,:,x) = conv_data_imag(:,:,x) + conv_bias_imag(x);
        if activation
            conv_data_real(:,:,x) = max(conv_data_real(:,:,x)  , 0);
            conv_data_imag(:,:,x) = max(conv_data_imag(:,:,x)  , 0);
        end
    end
end

