function res = fft2c(x)
fctr = size(x,1)*size(x,2);
res = zeros(size(x));
for l3 = 1:size(x,3)
    for l4 =1:size(x,4)
        res(:,:,l3,l4) = 1/sqrt(fctr)*fftshift(fft2(ifftshift(x(:,:,l3,l4))));
    end
end
end


