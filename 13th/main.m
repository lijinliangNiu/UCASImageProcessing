clc;
clear;
close all;

img = imread('lena512color.tiff');
img_gray = rgb2gray(img);
colormap(gray);
subplot(1,3,1);
imshow(img_gray);
title('灰度图');

%添加随机高斯白噪声
noise= wgn(512, 512, 22);  
img_noise = double(img_gray) + noise; 
subplot(1,3,2);
imshow(uint8(img_noise));
title('加入噪声');

[ll1, lh1, hl1, hh1]= wiener(img_noise, 'bior2.2');
[ll2, lh2, hl2, hh2]= wiener(ll1, 'bior2.2');
[ll3, lh3, hl3, hh3]= wiener(ll2, 'bior2.2');
[ll4, lh4, hl4, hh4]= wiener(ll3, 'bior2.2');
[ll5, lh5, hl5, hh5]= wiener(ll4, 'bior2.2');

ll4 = idwt2(ll5, lh5, hl5, hh5, 'bior2.2', size(ll4));
ll3 = idwt2(ll4, lh4, hl4, hh4, 'bior2.2', size(ll3));
ll2 = idwt2(ll3, lh3, hl3, hh3, 'bior2.2', size(ll2));
ll1 = idwt2(ll2, lh2, hl2, hh2, 'bior2.2', size(ll1));
res = idwt2(ll1, lh1, hl1, hh1, 'bior2.2');

res= uint8(res);
subplot(1, 3, 3);
imshow(res); 
title('重建图像');