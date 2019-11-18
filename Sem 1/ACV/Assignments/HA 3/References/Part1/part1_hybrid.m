clc;
clear all;
close all;
% im = imread('C:\Users\rusha\Desktop\proj_3\data\image1.jpg');
image5 = im2single(imread('bird.bmp'));
image6 = im2single(imread('plane.bmp'));
cutoff_frequency3 = 7;
filter3 = fspecial('Gaussian', cutoff_frequency3 * 4 + 1, cutoff_frequency3);

low_frequencies_bird = filter_my(image5, filter3);
high_frequencies_plane= image6 - filter_my(image6, filter3);
im = low_frequencies_bird + high_frequencies_plane;
im = rgb2gray(im);
subplot(2,2,1), imshow(im)


f_im = fft2(im);
F = fftshift(f_im); % Center FFT
F = abs(F); % Get the magnitude
F = log(F+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
F = mat2gray(F); % Use mat2gray to scale the image between 0 and 1
subplot(2,2,2),imshow(F); % Display the result

[M,N] = size(im);
H = lpfilter('gaussian',M,N,7);
filtf_im = f_im.*H;


F = fftshift(filtf_im); % Center FFT
F = abs(F); % Get the magnitude
F = log(F+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
F = mat2gray(F); % Use mat2gray to scale the image between 0 and 1
subplot(2,2,3),imshow(F); % Display the result
filt_im = ifft2(filtf_im);
subplot(2,2,4),imshow((filt_im));

figure
subplot(2,2,1), imshow(im)


f_im = fft2(im);
F = fftshift(f_im); % Center FFT
F = abs(F); % Get the magnitude
F = log(F+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
F = mat2gray(F); % Use mat2gray to scale the image between 0 and 1
subplot(2,2,2),imshow(F); % Display the result


H = hpfilter('gaussian',M,N,7);
filtf_im = f_im.*H;
F = fftshift(filtf_im); % Center FFT
F = abs(F); % Get the magnitude
F = log(F+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
F = mat2gray(F); % Use mat2gray to scale the image between 0 and 1
subplot(2,2,3),imshow(F); % Display the result

filt_im = ifft2(filtf_im);
subplot(2,2,4),imshow((filt_im));





image5 = im2single(imread('bird.bmp'));
image6 = im2single(imread('plane.bmp'));
cutoff_frequency3 = 11;
filter3 = fspecial('Gaussian', cutoff_frequency3 * 4 + 1, cutoff_frequency3);

low_frequencies_bird = filter_my(image5, filter3);
high_frequencies_plane= image6 - filter_my(image6, filter3);
im = low_frequencies_bird + high_frequencies_plane;
im = rgb2gray(im);
figure;
subplot(2,2,1), imshow(im)


f_im = fft2(im);
F = fftshift(f_im); % Center FFT
F = abs(F); % Get the magnitude
F = log(F+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
F = mat2gray(F); % Use mat2gray to scale the image between 0 and 1
subplot(2,2,2),imshow(F); % Display the result

[M,N] = size(im);
H = lpfilter('gaussian',M,N,7);
filtf_im = f_im.*H;


F = fftshift(filtf_im); % Center FFT
F = abs(F); % Get the magnitude
F = log(F+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
F = mat2gray(F); % Use mat2gray to scale the image between 0 and 1
subplot(2,2,3),imshow(F); % Display the result
filt_im = ifft2(filtf_im);
subplot(2,2,4),imshow((filt_im));

figure
subplot(2,2,1), imshow(im)


f_im = fft2(im);
F = fftshift(f_im); % Center FFT
F = abs(F); % Get the magnitude
F = log(F+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
F = mat2gray(F); % Use mat2gray to scale the image between 0 and 1
subplot(2,2,2),imshow(F); % Display the result


H = hpfilter('gaussian',M,N,7);
filtf_im = f_im.*H;
F = fftshift(filtf_im); % Center FFT
F = abs(F); % Get the magnitude
F = log(F+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
F = mat2gray(F); % Use mat2gray to scale the image between 0 and 1
subplot(2,2,3),imshow(F); % Display the result

filt_im = ifft2(filtf_im);
subplot(2,2,4),imshow((filt_im));
