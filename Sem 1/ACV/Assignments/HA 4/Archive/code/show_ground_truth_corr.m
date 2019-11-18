function show_ground_truth_corr()

%image1 = imread('../data/Notre Dame/921919841_a30df938f2_o.jpg');
%image2 = imread('../data/Notre Dame/4191453057_c86028ce1f_o.jpg');

%corr_file = '../data/Notre Dame/921919841_a30df938f2_o_to_4191453057_c86028ce1f_o.mat';

%image1 = imread('../data/Notre Dame/colors1.jpg');
%image2 = imread('../data/Notre Dame/colors2.jpg');

%corr_file = '../data/Notre Dame/colors1_colors2.mat';

%image1 = imread('../data/Notre Dame/bridge1.jpg');
%image2 = imread('../data/Notre Dame/bridge2.jpg');

%corr_file = '../data/Notre Dame/bridge1_bridge2.mat';

load(corr_file)

show_correspondence(image1, image2, x1, y1, x2, y2)