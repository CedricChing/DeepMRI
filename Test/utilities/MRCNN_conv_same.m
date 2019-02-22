function im_h = MRCNN_conv_same(model, im_b)
%% load CNN model parameters
load(model);
[hei, wid, ~] = size(im_b);
[conv1_channels,conv1_patchsize2,conv1_filters] = size(weights_conv1);
conv1_patchsize = sqrt(conv1_patchsize2);
fmap1_size = [hei-conv1_patchsize+1, wid-conv1_patchsize+1];
[conv2_channels,conv2_patchsize2,conv2_filters] = size(weights_conv2);
conv2_patchsize = sqrt(conv2_patchsize2);
fmap2_size = [fmap1_size(1)-conv2_patchsize+1, fmap1_size(2)-conv2_patchsize+1];
[conv3_channels,conv3_patchsize2] = size(weights_conv3);
conv3_patchsize = sqrt(conv3_patchsize2);
fmap3_size = [fmap2_size(1)-conv3_patchsize+1, fmap2_size(2)-conv3_patchsize+1];

%% conv1
conv1_data = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    for j = 1 : conv1_channels
        conv1_subfilter = reshape(weights_conv1(j,:,i), conv1_patchsize, conv1_patchsize);
        %conv1_data(:,:,i) = conv1_data(:,:,i) + convn(im_b(:,:,j), double(conv1_subfilter),'valid');
        conv1_data(:,:,i) = conv1_data(:,:,i) + imfilter(im_b(:,:,j), double(conv1_subfilter), 'same', 'replicate');
    end
    conv1_data(:,:,i) = max(conv1_data(:,:,i) + biases_conv1(i), 0);
end

%% conv2
conv2_data = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        conv2_subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        %conv2_data(:,:,i) = conv2_data(:,:,i) + convn(conv1_data(:,:,j), double(conv2_subfilter),'valid');
        conv2_data(:,:,i) = conv2_data(:,:,i) + imfilter(conv1_data(:,:,j), double(conv2_subfilter), 'same', 'replicate');
    end
    conv2_data(:,:,i) = max(conv2_data(:,:,i) + biases_conv2(i), 0);
end

%% conv3
conv3_data = zeros(hei, wid);
for i = 1 : conv3_channels
    conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    %conv3_data(:,:) = conv3_data(:,:) + convn(conv2_data(:,:,i), double(conv3_subfilter),'valid');
    conv3_data(:,:) = conv3_data(:,:) + imfilter(conv2_data(:,:,i), double(conv3_subfilter), 'same', 'replicate');
end

%% SRCNN reconstruction
im_h = conv3_data(:,:) + biases_conv3;
