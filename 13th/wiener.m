clc;
clear;
close all;

img = imread('lena512color.tiff');
% subplot(3,2,1);
% imshow(I);
% title('Ô­Ê¼Í¼Ïñ');
gray = rgb2gray(img);
% subplot(3,2,2);
% imshow(J);
colormap('gray');
% title('»Ò¶ÈÍ¼Ïñ');
img_noise=imnoise(gray, 'salt & pepper', 0.045); 
subplot(2, 1, 1);
imshow(img_noise);

[cA, cH, cV, cD] = dwt2(img_noise, 'bior4.4');

sigma_n = median(abs(cA(:))) / 0.6745;
sigmaA = mean(cA(:).^ 2) - sigma_n ^ 2;
sigmaH = mean(cH(:).^ 2) - sigma_n ^ 2;
sigmaV = mean(cV(:).^ 2) - sigma_n ^ 2;
sigmaD = mean(cD(:).^ 2) - sigma_n ^ 2;

cA_Y = sigmaA ^2 * cA/ ( sigmaA ^2 + sigma_n ^ 2);
cH_Y = sigmaH ^2 * cH/ ( sigmaH ^2 + sigma_n ^ 2);
cV_Y = sigmaV ^2 * cV/ ( sigmaV ^2 + sigma_n ^ 2);
cD_Y = sigmaD ^2 * cD/ ( sigmaD ^2 + sigma_n ^ 2);

XX = idwt2(cA_Y, cH_Y, cV_Y, cD_Y, 'bior4.4');
XX= uint8(XX);
subplot(2, 1, 2);
imshow(XX); 



