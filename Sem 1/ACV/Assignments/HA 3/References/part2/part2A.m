
close all; % closes all figures

%% Setup
% read images and convert to floating point format
img = im2single(imread('image1.jpg'));
img = rgb2gray(img);
[im in] = size(img);
template = im2single(imread('template.jpg'));
template = rgb2gray(template);
[fm fn] = size(template);


actual_X = 132;
actual_Y = 80;


ZMC_var =  zmc_f(img, template);
[a b] = max(ZMC_var(:));
[y, x] = ind2sub(size(ZMC_var),b);
ZMC_error = sqrt((actual_X - x).^2+(actual_Y - y).^2);

figure(1); subplot(2,2,1),imshow(ZMC_var,[]),title('zero mean correlation');

SSD_var = ssd_f(img,template);
[a b] = max(SSD_var(:));
[y, x] = ind2sub(size(SSD_var),b);
SSD_error = sqrt((actual_X - x).^2+(actual_Y - y).^2);

subplot(2,2,2),imshow(SSD_var,[]),title('sum square difference');;

NCC = normxcorr2(template,img);

NC = NCC(floor((fm/2)+1):(fm/2)+im,floor((fn/2)+1):(fn/2)+in);
[a b] = max(NC(:));
[y, x] = ind2sub(size(NC),b);
NC_error = sqrt((actual_X - x).^2+(actual_Y - y).^2);

subplot(2,2,3), imshow(NC,[]),title('normalized cross correlation');

fprintf('Error from ZMC - %f\n',ZMC_error);
fprintf('Error from SSD - %f\n',SSD_error);
fprintf('Error from NC  - %f\n',NC_error);
