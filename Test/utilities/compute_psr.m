function psr = compute_psr(label, x)
p = compute_psnr(label,x);
s = ssim(label,x);
r = compute_rmse(label,x);
psr = [p s r];