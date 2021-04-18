% LDA_run.m
% Classify new images using the training Datasets

dim = 400;  % square images (dim x dim)
tic;

% Format file names 
f = readtable('file_index.xlsx','Range', 'E1:E13'); 
% f = readtable('file_index.xlsx','Range', 'E1:E17'); 
f = table2array(f);
f = cell2mat(f); 

for i=1:size(f)
    t_data{i} = imread(f(i,:));          
    t_data{i} = t_data{i}(:,:,1);
    
    % Reshape each image into 1 column 
    T(:,i) = reshape(t_data{i}, dim*dim, 1); % (dim^2)x(#files) array
end

T = double(T);
class = LDA_faces(T,0);
true_class = [1 1 1 2 2 2 3 3 3 4 4 4]; % correct classification
disp("LDA Classification:");
disp(class);
disp("Actual Class:");
disp(true_class);    

time = toc;
% Calculate Accuracy
correct = (class==true_class); 
percent_correct = sum(correct)/length(correct)*100;
str = sprintf('Classification Accuracy:  %0.1f%%', percent_correct);
disp(str);
str2 = sprintf('Runtime: %0.1f seconds',time);
disp(str2);
