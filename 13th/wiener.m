function [ll, lh_denoise, hl_denoise, hh_denoise] = wiener(img, wname)
%wiener滤波并返回滤波后的4张图
[ll, lh, hl, hh]= dwt2(img, wname);
sigma_n = (median(abs(hh(:))) / 0.6745) ^ 2;

sigma_lh = mean(lh(:).^ 2) - sigma_n;
sigma_hl = mean(hl(:).^ 2) - sigma_n;
sigma_hh = mean(hh(:).^ 2) - sigma_n;

lh_denoise = (sigma_lh/ ( sigma_lh + sigma_n)) * lh;
hl_denoise = (sigma_hl/ ( sigma_hl + sigma_n)) * hl;
hh_denoise = (sigma_hh/ ( sigma_hh + sigma_n)) * hh;
end