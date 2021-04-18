% LDA_faces.m
% Fisher's Linear Discriminant Analysis

% NOTE: all ___.tif images, and file_index.xlsx 
% need to be in the same folder as the matlab script & matlab function

% LDA Classification function
% INPUT:  Y = new unclassified data (dim^2)x(#test images)
%         p = toggle for plotting Dataset projections 
%             and displaying median values
% OUTPUT: class = Face Shape Classification for each test image
%         (1=square, 2=round, 3=oval, 4=heart)

function class = LDA_faces(Y,p)
dim = 400;    % square image dimensions (dim x dim)
% Format file names 
    % square faces
    s = readtable('file_index.xlsx','Range', 'A1:A12');   % 11 square files
    s = table2array(s); s = cell2mat(s);   
    % round faces
    r = readtable('file_index.xlsx','Range', 'B1:B12');   % 11 round files
    r = table2array(r); r = cell2mat(r);  
    % oval faces
    o = readtable('file_index.xlsx','Range', 'C1:C12');   % 11 oval files
    o = table2array(o); o = cell2mat(o);  
    % heart faces
    h = readtable('file_index.xlsx','Range', 'D1:D12');   % 11 heart files
    h = table2array(h); h = cell2mat(h);
    
% Load training data 
    % load square images
    for i=1:size(s)
        s_data{i} = imread(s(i,:));        
        s_data{i} = s_data{i}(:,:,1);
        % Reshape each image into 1 column
        S(:,i) = reshape(s_data{i}, dim*dim, 1);  % (dim^2)x(#files) matrix
    end
    % load round images 
    for i=1:size(r)
        r_data{i} = imread(r(i,:));         
        r_data{i} = r_data{i}(:,:,1); 
        % Reshape each image into 1 column
        R(:,i) = reshape(r_data{i}, dim*dim, 1);  % (dim^2)x(#files) matrix
    end
    % load oval images 
    for i=1:size(o)
        o_data{i} = imread(o(i,:));        
        o_data{i} = o_data{i}(:,:,1);   
        % Reshape each image into 1 column
        O(:,i) = reshape(o_data{i}, dim*dim, 1);  % (dim^2)x(#files) matrix
    end
    % load heart images 
    for i=1:size(h)
        h_data{i} = imread(h(i,:));          
        h_data{i} = h_data{i}(:,:,1);   
        % Reshape each image into 1 column
        H(:,i) = reshape(h_data{i}, dim*dim, 1);  % (dim^2)x(#files) matrix
    end
    
% Format data and Compute Mean of each class
    S = double(S);
    R = double(R);
    O = double(O);
    H = double(H);
    m1 = sum(S,2)/length(s);     % mean of class Square
    m2 = sum(R,2)/length(r);     % mean of class Round
    m3 = sum(O,2)/length(o);     % mean of class Oval
    m4 = sum(H,2)/length(h);     % mean of class Heart
    
% Perform LDA (compute projection vector, and median) for each pair (6)
    %{
    Computed median values from training data:
    median1 = 1.7226e+06
    median2 = -3.1366e+09
    median3 = -1.2781e+08
    median4 = -1.2002e+08
    median5 = -3.3802e+09
    median6 = -6.3079e+08
    %}
    
    % (Square vs Round) (1v2)(w1)
    [w1,med1] = proj_vect_med(S,R,m1,m2,1,2,p);

    % (Oval vs Heart)   (3v4)(w2)
    [w2,med2] = proj_vect_med(O,H,m3,m4,3,4,p);

    % (Square vs Oval)  (1v3)(w3)
    [w3,med3] = proj_vect_med(S,O,m1,m3,1,3,p);

    % (Square vs Heart) (1v4)(w4)
    [w4,med4] = proj_vect_med(S,H,m1,m4,1,4,p);

    % (Round vs Oval)   (2v3)(w5)
    [w5,med5] = proj_vect_med(R,O,m2,m3,2,3,p);

    % (Round vs Heart)  (2v4)(w6)
    [w6,med6] = proj_vect_med(R,H,m2,m4,2,4,p);
%==========================================================================

% CLASSIFIER that takes in a (dim^2)x(# test images) array
% Each test image goes thru 2 rounds to be classified
    class = zeros(1,size(Y,2));
    for i=1:size(Y,2)
        y = Y(:,i);
    % ROUND 1:
        % 1 vs 2 (w1)
        [r1a] = competition(w1,med1,y,1,2);
        % 3 vs 4 (w2)
        [r1b] = competition(w2,med2,y,3,4);
        
    % ROUND 2:
        if      r1a==1 && r1b==3    % 1 vs 3 (w3)
            [c] = competition(w3,med3,y,r1a,r1b);
        elseif  r1a==1 && r1b==4    % 1 vs 4 (w4)
            [c] = competition(w4,med4,y,r1a,r1b);
        elseif  r1a==2 && r1b==3    % 2 vs 3 (w5)
            [c] = competition(w5,med5,y,r1a,r1b);
        else %  r1a==2 && r1b==4      2 vs 4 (w6)
            [c] = competition(w6,med6,y,r1a,r1b);
        end
        
        class(i)=c;
    end
end

function [winner] = competition(w,med,img,i,j)
%{ 
INPUT:  w (projection vector)
        med (median between projected datasets)
        img (test image)
        i (class 1)
        j (class 2)
OUTPUT: winner (class the image matches to best)
%}
    AL = [0 1 3 4;     % Class on Left side of median (dataset projection)
          1 0 3 4;
          3 3 0 4;
          4 4 4 0];
    AR = [0 2 1 1;     % Class on Right side of median (dataset projection)
          2 0 2 2;
          1 2 0 3;
          1 2 3 0];
     % Determine which class is on the left and right of the median (separation value)
     L = AL(i,j);      % left class
     R = AR(i,j);      % right class
     
     % Project test image onto real line to determine class 
     if (w'*img < med)  % if projection < median
         winner = L;    % then winner is Left class
     else 
         winner = R;    % else winner is Right class
     end
end

function [w,med] = proj_vect_med(D1,D2,mu1,mu2,i,j,p)
%{
INPUT:
    D1:  dataset 1
    D2:  dataset 2
    mu1: average (mean) image of D1
    mu2: average (mean) image of D2
    p:   toggle (if p==1, then plot projections of D1 & D2)

OUTPUT:
    w:   projection vector
    med: median (to separate projections of datasets)
%}
    x = cat(2,D1,D2);                   % matrix values from both datasets
    M = x'*(mu1-mu2);
    Sb = M*(M');                        % between class scatter
    M = x'*[D1(:,1:end)-mu1, D2(:,1:end)-mu2];
    Sw = M*(M');                        % within class scatter
    [V,d] = eig(Sb,Sw);
    id = find(diag(d)==max(diag(d)));   % biggest eigenvalue
    
    % w = x*eigenvector corresponding to largest eigenvalue  
    w = x*V(:,id);  
    
    % Project the D1 and D2 image data onto a real line (horiz. line y=1)
    p1_proj = D1'*w;
    p2_proj = D2'*w;
    MIN = min(min(p1_proj,p2_proj));
    MAX = max(max(p1_proj,p2_proj));
    med = 0.5*(MIN+MAX);
    
    if p==1
        figure;
        plot(p1_proj,ones(length(p1_proj),1),'r*'); hold on;   % class 1
        plot(p2_proj,ones(length(p2_proj),1),'bo'); hold on;   % class 2
        plot([MIN-2, MAX+2],ones(2,1),'k-'); hold on;     % horiz. line y=1
        plot(med*ones(3,1), [0 1 2]', 'c--');     % vert. line x=median
        legend(num2str(i), num2str(j), '$\mathbf{w}$', 'median', 'Interpreter','latex');
        title("Fisher's LDA");
        disp(med);
    end
end




