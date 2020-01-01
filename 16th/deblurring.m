clc;
clear;
img = imread('test.png');

% 倒谱
F = fft2(img);
Cep = ifft2(log(abs(F)));
Cep_c = fftshift(Cep);

% 求模糊核参数L theta
% 先求2个对称的极小值点
Cep_cl = Cep_c;
Cep_cl(:, 257:end) = 1;
[x(1), y(1)] = find(Cep_cl == min(min(Cep_cl)));

Cep_cr = Cep_c;
Cep_cr(:, 1:256)=1;
[x(2), y(2)] = find(Cep_cr == min(min(Cep_cr)));

L = sqrt((x(1)-x(2))^2+(y(1)-y(2))^2)/2;
theta = -ceil(atan((x(2)-x(1))/(y(2)-y(1)))*180/pi);

% 点扩散函数
PSF = fspecial('motion', L, theta);

%维纳滤波 并去振铃
wnr = deconvwnr(img, PSF, 0.01);
wnr = edgetaper(wnr, PSF);

% 算峰值信噪比
GT = rgb2gray(imread('lena512color.tiff'));
fprintf('Blured: %.2f\n',psnr(img,GT));
fprintf('Restored: %.2f\n',psnr(wnr,GT));

figure;
subplot(1,4,1);
imshow(img);
title('原图');

subplot(1,4,2);
imshow(log(abs(Cep_c)),[]);
title('倒谱');

subplot(1,4,3);
imshow(PSF,[]);
title('点扩散函数');

subplot(1,4,4);
imshow(wnr);
title('复原图像');
