%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab Demos
% (adapted from http://www.stanford.edu/class/cs223b/matlabIntro.html)
%
% Stefan Roth <roth (AT) cs DOT brown DOT edu>, 09/10/2003
% Alex Balan  <alb  (AT) cs DOT brown DOT edu>, 09/14/2004
% Leonid Sigal<ls   (AT) cs DOT brown DOT edu>, 09/14/2005
%
% Last modified: 09/14/2004
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Assignment 0 solutions
img = imread('cit.png');

mean(mean(img))
sum(sum(img)) / prod(size(img))
mean(img(:))                        % The colon operator here arranges 
                                    % elements of a matrix into a vector
mean2(img)

min(min(img))
min(img(:))                         % Recommended


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) Basics

% You can evaluate a portion of code by selecting it and pressing F9.

help general

% The symbol "%" is used to indicate a comment (for the remainder of
% the line).

% When writing a long Matlab statement that becomes to long for a
% single line use "..." at the end of the line to continue on the next
% line.  E.g.

A = [1, 2; ...
     3, 4];

% A semicolon at the end of a statement means that Matlab will not
% display the result of the evaluated statement. If the ";" is omitted
% then Matlab will display the result.  This is also useful for
% printing the value of variables, e.g.

A

% Matlab's command line is a little like a standard shell:
% - Use the up arrow to recall commands without retyping them (and 
%   down arrow to go forward in the command history).  
% - In linux, C-a moves to beginning of line (C-e for end), C-f moves 
%   forward a character and C-b moves back (equivalent to the left and 
%   right arrow keys), C-d deletes a character, C-k deletes the rest of 
%   the line to the right of the cursor, C-p goes back through the
%   command history and C-n goes forward (equivalent to up and down
%   arrows), Tab tries to complete a command.
% You can execute a linux command in matlab by prepending "!" to it
!ls -l
!rm *~

% Simple debugging:
% If the command "dbstop if error" is issued before running a script
% or a function that causes a run-time error, the execution will stop
% at the point where the error occurred. Very useful for tracking down
% errors.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(8) Working with (gray level) images

I = imread('cit.png');         % Read a PNG image

figure
imagesc(I)                    % Display it as gray level image
colormap gray;
axis image                    % Preserves image aspect ratio
colorbar                      % Turn on color bar on the side
pixval                        % Display pixel values interactively
truesize                      % Display at resolution of one screen
                              %   pixel per image pixel
truesize(2*size(I))           % Display at resolution of two screen
                              %   pixels per image pixel

I2 = imresize(I, 0.5, 'bil'); % Resize to 50% using bilinear 
                              %   interpolation
imagesc(I2)  
truesize  

imwrite(I2, 'test.png')       % Save image as PNG

help images                   % Lists Image Processing Toolbox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(7) Plotting 

x = [0 2 1 3 4]             % Basic plotting
plot(x);                     % Plot x versus its index values
pause                        % Wait for key press

figure;                      % Open new figure
x = pi*[-1:1/24:1]           % Vector of arguments for functions to be ploted
plot(x, sin(x));
xlabel('radians');           % Assign label for x-axis
ylabel('sin value');         % Assign label for y-axis
title('dummy title');        % Assign plot title

figure(1);                   % Use the figure with handle 1   
subplot(1, 2, 1);            % Multiple functions in separate graphs
plot(x, sin(x));             %   (see "help subplot")
axis square;                 % Make visible area square
subplot(1, 2, 2);
plot(x, 2*cos(x));
axis square;

figure(2);                   % Use the figure with handle 2   
plot(x, sin(x));
hold on;                     % Multiple functions in single graph           
plot(x, 2*cos(x), '--');     % '--' chooses different line pattern
legend('sin', 'cos');        % Assigns names to each plot
hold off;                    % Stop putting multiple figures in current
                             %   graph
pause
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% (2) Basic types in Matlab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (A) The basic types in Matlab are scalars (usually double-precision
% floating point), vectors, and matrices:

N = 5                        % A scalar

A = [1 2; 3 4]               % Creates a 2x2 matrix
B = [1,2; 3,4]               % The simplest way to create a matrix is
                             % to list its entries in square brackets.
                             % The ";" symbol separates rows;
                             % the (optional) "," separates columns.

v = [1 0 0]                  % A row vector
v = [1; 2; 3]                % A column vector
v = v'                       % Transpose a vector (row to column or 
                             %   column to row)
v = 1:.5:3                   % A vector filled in a specified range: 
v = pi*[-4:4]/4 + 2          %   [start:stepsize:end]
                             %   (brackets are optional)
v = []                       % Empty vector


b = zeros(3, 2)             % Matrix of zeros
c = ones(2, 3)              % Matrix of ones
d = eye(4)                  % Identity matrix
d2 = diag([1:4]/4)          % Diagonal matrix
e = rand(64, 128);          % Random matrix (uniform) between [0,1]
f = randn(64, 128);         % Random matrix (normal) N(0,1)
g = zeros(128);             % Matrix of zeros (128x128)

subplot(2,3,1); imagesc(b, [0 1]); colormap gray; axis image;
subplot(2,3,2); imagesc(c, [0 1]); colormap gray; axis image;
subplot(2,3,3); imagesc(d2, [0 1]); colormap gray; axis image;
subplot(2,3,4); imagesc(e, [0 1]); colormap gray; axis image;
subplot(2,3,5); imagesc(f, [-3 3]); colormap gray; axis image;
subplot(2,3,6); imagesc(g, [0 1]); colormap gray; axis image;

pause; close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (C) Indexing vectors and matrices.
% Warning: Indices always start at 1 and *NOT* at 0!

v = [1 2 3]
v(3)                         % Access a vector element 


m = [1 2 3 4; 5 7 8 8; 9 10 11 12; 13 14 15 16]
m(3, 2)                      % Access a matrix element
                             %       matrix(ROW #, COLUMN #)

m(3:-1:1, [1 3])           
                             
p = imread('peppers256.tif');
b = p(68:148, :);             % Access whole matrix rows
c = p(:, 68:148);             % Access whole matrix columns
d = p(32:48, 16:32);         % Access some elements
e = p(1:2:32, :);            
f = p(:, 1:2:end);           % Keyword "end" accesses the remainder of a
                             %   column or row

subplot(2,3,1); imagesc(p, [0 255]); colormap gray; axis image;
subplot(2,3,2); imagesc(b, [0 255]); colormap gray; axis image;
subplot(2,3,3); imagesc(c, [0 255]); colormap gray; axis image;
subplot(2,3,4); imagesc(d, [0 255]); colormap gray; axis image;
subplot(2,3,5); imagesc(e, [0 255]); colormap gray; axis image;
subplot(2,3,6); imagesc(f, [0 255]); colormap gray; axis image;

pause; close;

m = rand(128,256);
size(m)                      % Returns the size of a matrix
size(m, 1)                   % Number of rows
size(m, 2)                   % Number of columns
length(m)                    % max(size(m))
m1 = zeros(size(m));         % Create a new matrix with the size of m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (3) Simple operations on vectors and matrices

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (A) Element-wise operations:

% These operations are done "element by element".  If two 
% vectors/matrices are to be added, subtracted, or element-wise
% multiplied or divided, they must have the same size.

% For element by element operations use: .+ .- .* ./ .^
% For algebraic/matrix operations use: + - * / ^

% Load some images into matrices, and convert the elements 
% from uint8 to double to allow math operations

p = imread('peppers256.tif');   
L = imread('lena.tif');
p = double(p);                % Convert from uint8 to double, to allow
L = double(L);                %   math operations

oldP = uint8(p);              % Convert back to uint8 for writing
imwrite(oldP, 'test.tif')     % Save image as PNG


% l is double the size than p
size(p)
size(L)

% make them the same
L = imresize(L,0.5,'bil');
size(L)

% normalize pixel range between [0 1]
p = p / max(p(:));
L = L / max(L(:));

b = 2 * p;                   % Scalar multiplication
c = p / 1.5;                 % Scalar division
d = p + L;                   % matrix addition
e = p - L;                   % matrix subtraction

subplot(2,3,1); imagesc(p, [0 2]); colormap gray; axis image;
subplot(2,3,2); imagesc(L, [0 2]); colormap gray; axis image;
subplot(2,3,3); imagesc(b, [0 2]); colormap gray; axis image;
subplot(2,3,4); imagesc(c, [0 2]); colormap gray; axis image;
subplot(2,3,5); imagesc(d, [0 2]); colormap gray; axis image;
subplot(2,3,6); imagesc(e, [-1 1]); colormap gray; axis image;

pause; close;

b = p .^ 2;                  % Element-wise squaring (note the ".")
c = p .* L;                  % Element-wise multiplication (note the ".")
d = p ./ d;                  % Element-wise division (note the ".")
e = log(p);                  % Element-wise logarithm
f = round(5 * p);            % Element-wise rounding to nearest integer

subplot(2,3,1); imagesc(p); colormap gray; axis image;
subplot(2,3,2); imagesc(b); colormap gray; axis image;
subplot(2,3,3); imagesc(c); colormap gray; axis image;
subplot(2,3,4); imagesc(d); colormap gray; axis image;
subplot(2,3,5); imagesc(e); colormap gray; axis image;
subplot(2,3,6); imagesc(f); colormap gray; axis image;

% Example ^ vs .^ :
subplot(1,2,1); imshow(p.^2, [0,max(max(p.^2))])
subplot(1,2,2); imshow(p ^2, [0,max(max(p ^2))])

pause; close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (D) Reshaping and assembling matrices:

a = [1 2; 3 4; 5 6]          % A 3x2 matrix
b = a(:)                     % Make 6x1 column vector by stacking 
                             %   up columns of a
sum(a(:))                    % Useful:  sum of all elements

a = reshape(L, 512, 128);    % Make 512x128 matrix out of vector 
                             %   elements (column-wise)
% Matrix concatenation
b = [p L];                   % Horizontal concatenation (see horzcat)

c = [p; L];                  % Vertical concatenation (see vertcat)

d = repmat(p, 3, 2);         % Create a 3x2 replication of the image p

subplot(2,2,1); imagesc(a, [0 1]); colormap gray; axis image;
subplot(2,2,2); imagesc(b, [0 1]); colormap gray; axis image;
subplot(2,2,3); imagesc(c, [0 1]); colormap gray; axis image;
subplot(2,2,4); imagesc(d, [0 1]); colormap gray; axis image;

pause; close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (B) Vector Operations
% Built-in Matlab functions that operate on vectors

a = [1 4 6 3]                % A row vector
sum(a)                       % Sum of vector elements
mean(a)                      % Mean of vector elements
var(a)                       % Variance of elements
std(a)                       % Standard deviation
max(a)                       % Maximum
min(a)                       % Minimum

% If a matrix is given, then these functions will operate on each column
%   of the matrix and return a row vector as result
a = [1 2 3; 4 5 6]           % A matrix
mean(a)                      % Mean of each column
max(a)                       % Max of each column    
max(max(a))                  % Obtaining the max of a matrix 
mean(a, 2)                   % Mean of each row (second argument specifies
                             %   dimension along which operation is taken)

% ' performs matrix transposition

[1 2 3] * [4 5 6]'           % 1x3 row vector times a 3x1 column vector
                             %   results in a scalar.  Known as dot product
                             %   or inner product.  Note the absence of "."

[1 2 3]' * [4 5 6]           % 3x1 column vector times a 1x3 row vector 
                             %   results in a 3x3 matrix.  Known as outer
                             %   product.  Note the absence of "."

% Other elementary functions:
% >> abs(x)       % absolute value of x
% >> exp(x)       % e to the x-th power
% >> fix(x)       % rounds x to integer towards 0
% >> log10(x)     % common logarithm of x to the base 10
% >> rem(x,y)     % remainder of x/y
% >> sqrt(x)      % square root of x
% >> sin(x)       % sine of x; x in radians
% >> acoth(x)     % inversion hyperbolic cotangent of x

% Other element-wise arithmetic operations include e.g. :
%   floor, ceil, ...

help elfun   % get a list of all available elementary functions


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Other data structures:

% "Structures" are multidimensional Matlab arrays with elements accessed 
% by textual field designators. For example,

A.name = 'Rudi';
A.score = [ 2 3 4];
A.fancy = rand;
A

% Like everything else in Matlab, structures are arrays, 
% so you can insert additional elements.
A(2) = A(1);
A(2).name = 'Fredi';
A(1)
A(2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% "Cell Arrays"
% Cell arrays in Matlab are multidimensional arrays whose elements can
% be anything. Cell arrays are created by enclosing a 
% miscellaneous collection of things in curly braces, fg:

S = 'Hello World!'
R = rand(2,3,2)

C = {R -1 S}

% To retrieve the contents of one of the cells, use subscripts in 
% curly braces.
C{3}

% Like numeric arrays also cell arrays can be mulitdimensional
CM{2,2} = S;
CM{1,1} = 'hallo';
CM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (4) Control statements & vectorization

% Syntax of control flow statements:
% 
% for VARIABLE = EXPR
%     STATEMENT
%      ...
%     STATEMENT
% end 
%
%   EXPR is a row vector here, e.g. 1:10 or -1:0.5:1 or [1 4 7]
%     - it should never be a column vector
% 
%
% while EXPRESSION
%     STATEMENTS
% end
% 
% if EXPRESSION
%     STATEMENTS 
% elseif EXPRESSION
%     STATEMENTS
% else
%     STATEMENTS
% end 
%
%   (elseif and else clauses are optional, the "end" is required)
%
%   EXPRESSIONs are usually made of relational clauses, e.g. a < b
%   The operators are <, >, <=, >=, ==, ~=  (almost like in C(++))
%
%   Element-wise Logical operators :
%     and: &
%     or: |
%     not: ~
%     exclusive or: xor
%   For short-circuit: &&, ||

% Warning:
%   Loops run very slowly in Matlab, because of interpretation overhead.
%   This has gotten somewhat better in version 6.5, but you should
%   nevertheless try to avoid them by "vectorizing" the computation, 
%   i.e. by rewriting the code in form of matrix operations.  This is
%   illustrated in some examples below.

R = [4 5] > [3 7]
any(R)                  % True if any element of a vector is nonzero
all(R)                  % True if all elements of a vector are nonzero


% Examples:
for i=1:2:7                  % Loop from 1 to 7 in steps of 2
  i                          % Print i
end

for i=[5 13 -1]              % Loop over given vector
  if (i > 10)                % Sample if statement
    disp('Larger than 10')   % Print given string
  elseif i < 0               % Parentheses are optional
    disp('Negative value') 
  else
    disp('Something else')
  end
end


% Here is another example: given an mxn matrix A and a 1xn 
% vector v, we want to subtract v from every row of A.

m = 50000; n = 10; A = ones(m, n); v = 2 * rand(1, n); 
%
% Implementation using loops:
tic                          % start stopwatch
for i=1:m
  A(i,:) = A(i,:) - v;
end
toc                          % stop stopwatch

% We can compute the same thing using only matrix operations
tic                          % start stopwatch
A = ones(m, n) - repmat(v, m, 1);   % This version of the code runs 
                                    %   much faster!!!
toc                          % stop stopwatch

% We can vectorize the computation even when loops contain
%   conditional statements.
%
% Example: given an mxn matrix A, create a matrix B of the same size
%   containing all zeros, and then copy into B the elements of A that
%   are greater than zero.

% Implementation using loops:
tic                          % start stopwatch
B = zeros(m,n);
for i=1:m
  for j=1:n
    if A(i,j)>0
      B(i,j) = A(i,j);
    end
  end
end
toc                          % stop stopwatch

% All this can be computed w/o any loop!
tic                          % start stopwatch
B = zeros(m,n);
ind = find(A > 0);           % Find indices of positive elements of A 
                             %   (see "help find" for more info)
B(ind) = A(ind);             % Copies into B only the elements of A
                             %   that are > 0
toc                          % stop stopwatch

% The indeces "find" returns can be converted back to row-column index
% using ind2sub.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fprintf : Can be used to print output to the screen or to a file. 

% Print to the screen
x=pi;
fprintf('\nThis is a test, x=%g \n',x)
% This command prints the number x in
% either exponential or fixed point formant
% (whichever is shorter) and then starts a new
% line.

disp(x)
disp(['X = ' num2str(x)])       % Display by string concatenation

% Print to a file called 'output.txt':
fid=fopen('output.txt')
fprintf(fid,'hello %g', x)

% Get input from the keyboard, e.g.,
a = input('Left Endpoint a = ?')     


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(6) Creating scripts or functions using m-files: 
%
% Matlab scripts are files with ".m" extension containing Matlab 
% commands.  Variables in a script file are global and will change the
% value of variables of the same name in the environment of the current
% Matlab session.  A script with name "script1.m" can be invoked by
% typing "script1" in the command window.

% Functions are also m-files. The first line in a function file must be
% of this form: 
% function [outarg_1, ..., outarg_m] = myfunction(inarg_1, ..., inarg_n)
%
% The function name should be the same as that of the file 
% (i.e. function "myfunction" should be saved in file "myfunction.m"). 
% Have a look at myfunction.m and myotherfunction.m for examples.
%
% Functions are executed using local workspaces: there is no risk of
% conflicts with the variables in the main workspace. At the end of a
% function execution only the output arguments will be visible in the
% main workspace.
 
a = [1 2 3 4];               % Global variable a
b = myfunction(a .^ 2)       % Call myfunction which has local 
                             %   variable a
a                            % Global variable a is unchanged

[c, d] = ...
  myotherfunction(a, b)      % Call myotherfunction with two return
                             % values

%    %%%In "myfunction.m" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    function y = myfunction(x)
%    % Function of one argument with one return value
%
%    a = [1 1 1 1];              % Have a global variable of the same name
%    y = a + x;
%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%    %%%In "myotherfunction.m" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    function [y, z] = myotherfunction(a, b)
%    % Function of two arguments with two return values
%
%    y = a + b;
%    z = a - b;
%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(5) Saving your work

who                          % List variables in workspace
whos                         % List variables w/ info about size, type, etc.

save myfile                  % Saves all workspace variables into
                             %   file myfile.mat
save myfile a b              % Saves only variables a and b

clear a b                    % Removes variables a and b from the
                             %   workspace
clear                        % Clears the entire workspace

load myfile a                % Loads variable a

load myfile                  % Loads variable(s) from myfile.mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
