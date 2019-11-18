function output = my_imfilter(image, filter,filter_type)
% This function is intended to behave like the built in function imfilter()
% See 'help imfilter' or 'help conv2'. While terms like "filtering" and
% "convolution" might be used interchangeably, and they are indeed nearly
% the same thing, there is a difference:
% from 'help filter2'
%    2-D correlation is related to 2-D convolution by a 180 degree rotation
%    of the filter matrix.

% Your function should work for color images. Simply filter each color
% channel independently.

% Your function should work for filters of any width and height
% combination, as long as the width and height are odd (e.g. 1, 7, 9). This
% restriction makes it unambigious which pixel in the filter is the center
% pixel.

% Boundary handling can be tricky. The filter can't be centered on pixels
% at the image boundary without parts of the filter being out of bounds. If
% you look at 'help conv2' and 'help imfilter' you see that they have
% several options to deal with boundaries. You should simply recreate the
% default behavior of imfilter -- pad the input image with zeros, and
% return a filtered image which matches the input resolution. 
% A better approach is to mirror the image content over the boundaries for 
% padding.

% % Uncomment if you want to simply call imfilter so you can see the desired
% % behavior. When you write your actual solution, you can't use imfilter,
% % filter2, conv2, etc. Simply loop over all the pixels and do the actual
% % computation. It might be slow.



%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%


img=imread(image);
%Convert to grayscale
img=rgb2gray(img); 
subplot(2,2,1), imshow(img)

%Determine good padding for Fourier transform
padd = paddedsize(size(img));

%Create a Gaussian Lowpass filter 5% the width of the Fourier transform
%D0 = 0.05*PQ(1);
D0 = 0.07*padd(1);
%D0 = 0.01*PQ(1);
if filter_type == 0
 filter_img = hpfilter(filter, padd(1), padd(2), D0);
elseif filter_type == 1
 filter_img = lpfilter(filter, padd(1), padd(2), D0);   
 end
% Calculate the discrete Fourier transform of the image
FT=fft2(double(img),size(filter_img,1),size(filter_img,2));

% Apply the filter to the Fourier spectrum of the image
FS_I = filter_img.*FT;

% convert the result to the spacial domain.
S_D=real(ifft2(FS_I)); 

% Crop the image to undo padding
S_D=S_D(1:size(img,1), 1:size(img,2));

%Display the image
subplot(2,2,2), imshow(S_D, [])

% Move the origin of the transform to the center of the frequency rectangle.
centered=fftshift(FT);
Centerd_Shift=fftshift(FS_I);
% use abs to compute the magnitude and use log to brighten display
O1=log(1+abs(centered)); 
O2=log(1+abs(Centerd_Shift));
subplot(2,2,3), imshow(O1,[])
subplot(2,2,4), imshow(O2,[])

output = filter_img;