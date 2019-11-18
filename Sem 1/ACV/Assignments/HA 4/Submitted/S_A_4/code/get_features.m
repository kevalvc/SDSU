% Local Feature Stencil Code
 
% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features] = get_features(image, x, y, feature_width)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one. This is not required,
% though.

% Placeholder that you can delete. Empty features.
number_of_points = size(x,1);
features = zeros(number_of_points, 128);

small_gaussian = fspecial('Gaussian', [feature_width feature_width], 1);
large_gaussian = fspecial('Gaussian', [feature_width feature_width], feature_width/2);

[gx, gy] = imgradientxy(small_gaussian);
ix = imfilter(image, gx);
iy = imfilter(image, gy);

get_octant_value = @(x,y) (ceil(atan2(y,x)/(pi/4)) + 4);

orients = arrayfun(get_octant_value, ix, iy);
magx = hypot(ix, iy);
c_size = feature_width/4;
for ii = 1:number_of_points
    range_frame_x = (x(ii) - 2*c_size): (x(ii) + 2*c_size-1);
    range_frame_y = (y(ii) - 2*c_size): (y(ii) + 2*c_size-1);
    frame_mag_val = magx(range_frame_y, range_frame_x);
    frame_mag_val = frame_mag_val.*large_gaussian;
    frame_orients = orients(range_frame_y, range_frame_x);
    
    % Looping through each cell in the frame
    for i = 0:3
        for j = 0:3
            cell_orients = frame_orients(i*4+1:i*4+4, j*4+1:j*4+4);
            cell_mag = frame_mag_val(i*4+1:i*4+4, j*4+1:j*4+4);
            for o = 1:8
                f = cell_orients == o;
                features(ii, (i*32 + j*8) + o) = sum(sum(cell_mag(f)));
            end
        end
    end
end
features = diag(1./sum(features,2))*features; %Normalize feature vectors
    
end












