close all


%% Setup
%Here the flag used is 0 for High Frequency Images and 1 for Low Frequency
%Images.

high_freq_image = my_imfilter('image1.jpg','gaussian',0);
%low_freq_image = my_imfilter('image2.jpg','gaussian',1);
clc;

