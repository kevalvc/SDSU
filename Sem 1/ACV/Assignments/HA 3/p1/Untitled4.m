img=imread(image);

%Convert to grayscale img=rgb2gray(img); subplot(2,2,1), imshow(img)



%Determine good padding for Fourier transform padd = paddedsize(size(img));



%Create a Gaussian Lowpass filter 5% the width of the Fourier transform




D0 = 0.07*padd(1);




if filter_type == 0 


filter_img = hpfilter(filter, padd(1), padd(2), D0);

elseif filter_type == 1

filter_img = lpfilter(filter, padd(1), padd(2), D0);

end

% Calculate the discrete Fourier transform of the image

FT=fft2(double(img),size(filter_img,1),size(filter_img,2));




% Apply the filter to the Fourier spectrum of the image

FS_I = filter_img.*FT;

% spacial domain. S_D=real(ifft2(FS_I)); S_D=S_D(1:size(img,1), 1:size(img,2)); subplot(2,2,2), imshow(S_D, [])
% Move the origin of the transform to the center of the frequency rectangle. centered=fftshift(FT);
Centerd_Shift=fftshift(FS_I);

% use abs to compute the magnitude and use log to brighten display

O1=log(1+abs(centered)); 
O2=log(1+abs(Centerd_Shift)); 
subplot(2,2,3), imshow(O1,[]) ;
subplot(2,2,4), imshow(O2,[]); 
output = filter_img;