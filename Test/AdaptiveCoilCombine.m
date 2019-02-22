function C_Img = AdaptiveCoilCombine(Img, Rn)
% Coil Combine according to DO Walsh et al. Adaptive Reconstruction of
% Phased Array MR Imagery, MRM, 2000.
% Created by Zou Chao 2013-11-18

% Img: 2D images of individual coil (row * column * TEs * coils)
% Rn: noise correlation matrix

n_dim = ndims(Img);
if n_dim == 4
    [n_row, n_col, n_TE, n_coil] = size(Img);
elseif n_dim == 3
    n_TE = 1;
    [n_row, n_col, n_coil] = size(Img);
    Img = reshape(Img,[n_row n_col n_TE n_coil]);
else
    disp('Invalid image data !')
    return;
end
C_Img = zeros(n_row, n_col, n_TE);
if nargin < 2,
    Rn = eye(n_coil);
end
K_size = 7;
iRn = inv(Rn);

%% Compute Rs
tic;
Rs = zeros(n_row, n_col, n_coil, n_coil);
% for n = 1:1:n_TE
%     for j = 1:1:n_coil
%         for k = 1:1:j - 1
% %             Rs(:, :, j, k) = Rs(:, :, j, k) + Img(:, :, n, j) .* conj(Img(:, :, n, k));
%             Rs(:, :, j, k) = Rs(:, :, j, k) + filter2(ones(K_size), Img(:, :, n, j) .* conj(Img(:, :, n, k)), 'same');        
%             Rs(:, :, k, j) = Rs(:, :, k, j) + conj(Rs(:, :, j, k));
%         end
% %         Rs(:, :, j, j) = Img(:, :, n, j) .* conj(Img(:, :, n, j));
%         Rs(:, :, j, j) = Rs(:, :, j, j) + filter2(ones(K_size), Img(:, :, n, j) .* conj(Img(:, :, n, j)), 'same');
%     end
% end
for j = 1:1:n_coil
    for k = 1:1:n_coil
        for n = 1:1:n_TE
            Rs(:,:,j,k) = Rs(:,:,j,k) + filter2(ones(K_size), Img(:, :, n, j) .* conj(Img(:, :, n, k)), 'same'); 
        end
    end
end
toc;

tic;

% for y = 1:1:n_row
%     for x = 1:1:n_col
%         %% Find the largest eigenvalue
%         [E S] = svd(iRn * squeeze(Rs(y, x, :, :)));
%         m = E(:, 1);
%         C_Img(y, x, :) = m' * squeeze(Img(y, x, :, :)).';
%     end
% end

v = ones(n_row, n_col, n_coil);
N = 2;
d = zeros(n_row, n_col);
for n = 1:1:N
    v=squeeze(sum(Rs.*repmat(v,[1 1 1 n_coil]),3)); 
    d=sqrt(sum(v.*conj(v), 3));
    d( d <= eps) = eps;
	v=v./repmat(d,[1 1 n_coil]);
end

v = repmat(v, [1 1 1 n_TE]);
v = permute(v, [1 2 4 3]);

C_Img = sum(Img .* v, 4);

toc;


