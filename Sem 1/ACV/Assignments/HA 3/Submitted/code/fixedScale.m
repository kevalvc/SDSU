
close all;

image = im2single(imread('E:\Data\SDSU\Sem 1\ACV\Assignments\HA 3\Submitted\data\elonmusk1.jpg'));
image = rgb2gray(image);
[im im2] = size(image);
template = im2single(imread('E:\Data\SDSU\Sem 1\ACV\Assignments\HA 3\Submitted\data\tempMusk.jpg'));
template = rgb2gray(template);
[t1 t2] = size(template);

ogg_X = 130;
ogg_Y = 282 ;

ZMC_varr =  zmc_func(image, template);
[m1 m2] = max(ZMC_varr(:));
[n1, n2] = ind2sub(size(ZMC_varr),m2);

Zmc_err = sqrt((ogg_X - n2).^2+(ogg_Y - n1).^2);

figure(1); subplot(2,2,1),imshow(ZMC_varr,[]),title('zero mean correlation');

SSD_varr = ssd_func(image,template);
[m1 m2] = max(SSD_varr(:));
[n1, n2] = ind2sub(size(SSD_varr),m2);

Ssd_err = sqrt((ogg_X - n2).^2+(ogg_Y - n1).^2);

subplot(2,2,2),imshow(SSD_varr,[]),title('sum square difference');

NCC_varr = normxcorr2(template,image);

NCC = NCC_varr(floor((t1/2)+1):(t1/2)+im,floor((t2/2)+1):(t2/2)+im2);
[m1 m2] = max(NCC(:));
[n1, n2] = ind2sub(size(NCC),m2);

Ncc_err = sqrt((ogg_X - n2).^2+(ogg_Y - n1).^2);

subplot(2,2,3), imshow(NCC,[]),title('normalized cross correlation');

fprintf('Error in ZMC is: %f\n',Zmc_err);
fprintf('Error in SSD is: %f\n',Ssd_err);
fprintf('Error in NCC is: %f\n',Ncc_err);
