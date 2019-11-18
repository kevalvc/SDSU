% Local Feature Stencil Code


% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the features as additional inputs.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features 1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.
function [matches, confidences] = match_features(features1, features2)

% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).

% For extra credit you can implement various forms of spatial verification of matches.


% 
%< Placeholder that you can delete. Random matches and confidences
%pleae detect this following paragraph and implement your own codes; 
%'matches':and 'confidences' according to the
%near neighbor based matching algorithm. 

%take a threshold value

threshold = 0.75;
distance_matrix = pdist2(features1, features2, 'euclidean');

[sorted_disancet_matrix, indices] = sort(distance_matrix, 2);
inverse_confidence = (sorted_disancet_matrix(:,1)./sorted_disancet_matrix(:,2));
confidences = 1./inverse_confidence(inverse_confidence < threshold);

matches = zeros(size(confidences,1), 2);
matches(:,1) = find(inverse_confidence < threshold);
matches(:,2) = indices(inverse_confidence < threshold, 1);

% We sort the matches so that the strongest onces are at the top.
[confidences, indic] = sort(confidences, 'descend');
matches = matches(indic,:);