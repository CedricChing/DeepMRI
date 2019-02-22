function res = mtimes(a,x)
% This method performs the SPIRiT operator

kernel = a.kernel;
nCoils = size(kernel,3);
kSize = size(kernel); kSize = kSize(1:2);
method = a.method;
%

switch method

case {'conv'}
	res = zeros(size(x));
    if a.adjoint
        for n=1:nCoils
            tmpk = kernel(:,:,:,n);
            tmpk(floor(kSize(1)/2)+1, floor(kSize(2)/2)+1,n) = ...
			tmpk(floor(kSize(1)/2)+1, floor(kSize(2)/2)+1,n) -1;
            res = res + adjSPIRiT(x(:,:,n),tmpk); 
        end
    else
        for n=1:nCoils
            tmpk = kernel(:,:,:,n);
            tmpk(floor(kSize(1)/2)+1, floor(kSize(2)/2)+1,n) = ...
		    tmpk(floor(kSize(1)/2)+1, floor(kSize(2)/2)+1,n) -1;
            res(:,:,n) = SPIRiT(x, tmpk);
        end

    end

case {'fft'}
	
	res = zeros(size(x));
    if a.adjoint
        
        xx = ifft2c(x);
        for n=1:nCoils
            tmpk = squeeze(conj(a.KERNEL(:,:,n,:)));
            res(:,:,n) = sum(tmpk.*xx,3); 
        end
        res = fft2c(res)-x;
    else
        xx = ifft2c(x);
        for n=1:nCoils
            tmpk = a.KERNEL(:,:,:,n);
            res(:,:,n) = sum(tmpk.*xx,3);
        end
        res = fft2c(res)-x;
    end
    
 case {'image'}
    res = zeros(size(x));
    if a.adjoint
        for n=1:nCoils
            tmpk = squeeze(conj(a.KERNEL(:,:,n,:)));
            res(:,:,n) = sum(tmpk.*x,3); 
        end
        res = res - x;
    else
        for n=1:nCoils
            tmpk = a.KERNEL(:,:,:,n);
            res(:,:,n) = sum(tmpk.*x,3); 
        end
        res = res-x;
    end
 case {'proj_fft'}
    %  adjoint is the same as forward. 
    res = zeros(size(x));
    xx = ifft2c(x);
    for n=1:size(a.KERNEL,4)
         tmpk = a.KERNEL(:,:,:,n);
         res = res + repmat(sum(conj(tmpk).*xx,3),[1,1,size(a.KERNEL,3)]).*tmpk;
    end
    res = fft2c(res) - x;
        
 case {'proj_image'}
    %  adjoint is the same as forward. 
    res = zeros(size(x));
    xx = ifft2c(x);
    for n=1:size(a.KERNEL,4)
         tmpk = a.KERNEL(:,:,:,n);
         res = res + repmat(sum(conj(tmpk).*xx,3),[1,1,size(a.KERNEL,3)]).*tmpk;
    end
    res = res - x;
        
           
end
