function res = ifft2c(x)
fctr = size(x,1)*size(x,2);
res = zeros(size(x));
for l3 = 1:size(x,3)
    for l4 =1:size(x,4)
        res(:,:,l3,l4) = sqrt(fctr)*fftshift(ifft2(ifftshift(x(:,:,l3,l4))));
    end
end
end
