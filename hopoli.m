%{
Fernando Lejarza, April 19th, 2015, fl14@rice.edu
hopoli.m

%} 

function hopoli 

% Reading training images
i = imread('obama.jpg'); 
ob = im2bw(i); % black and white mtx 
ob1 = ob(1:900,300:1200); % croping image
ob2 = ob1(1:5:end-1,1:5:end); % coarsening image
size_ob = size(ob2) ; 


i = imread('trump.jpg'); 
tr = im2bw(i); 
disp(size(tr))
tr1 = tr(50:1200,300:1500);% croping 
tr1 = imresize(tr1,[900,901]) ;
tr2 = tr1(1:5:end-1,1:5:end); % 

i = imread('bernie.jpg'); 
br = im2bw(i);
br1 = imresize(br,[900,901]) ;
br2 = br1(1:5:end-1,1:5:end);

i = imread('warren.jpg');
wr = im2bw(i); 
wr1 = wr(50:1230,298:end-500); 
wr1 = imresize(wr1,[900,901]) ;
wr2 = wr1(1:5:end-1,1:5:end); 

figure(1)
hold on
subplot(2,2,1); imshow(ob2); title('Training Pattern I'); 
subplot(2,2,2); imshow(tr2); title('Training Pattern II');
subplot(2,2,3); imshow(br2); title('Training Pattern III'); 
subplot(2,2,4); imshow(wr2); title('Training Pattern IV');


P1 = real([reshape(ob2,size_ob(1)*size_ob(2),1) reshape(tr2,size_ob(1)*size_ob(2),1) ...
    reshape(br2,size_ob(1)*size_ob(2),1) reshape(wr2,size_ob(1)*size_ob(2),1)]);


P = 2*P1 -1; % P matrix containing all 1s and -1s
W = P*P'; % Synaptic weights matrix

figure(2) 
imagesc(W) ;colorbar; axis off; title('The Hopefield Weight Matrix');

% Distorting training images to generate test data
PN = makesomenoise(P); 


err = 1; 

ns = [reshape(PN(:,1),size_ob(1),size_ob(2)) reshape(PN(:,2),size_ob(1),size_ob(2))... 
 reshape(PN(:,3),size_ob(1),size_ob(2)) reshape(PN(:,4),size_ob(1),size_ob(2))]; 

 
s = PN; % initializing actuall state 

while err>0 % Loop used to get outputs considering the error (difference 
    % between the atual state and the new state 

    ns = sign2(W*s); 

    err = max(abs(s-ns)); 
 
    s = ns; 

end

figure(3) 
subplot(2,4,1); imshow(reshape(PN(:,1),size_ob(1),size_ob(2))); title('Input Pattern');
subplot(2,4,2); imshow(reshape(s(:,1),size_ob(1),size_ob(2))); title('Output Pattern');
subplot(2,4,3); imshow(reshape(PN(:,2),size_ob(1),size_ob(2))); title('Input Pattern');
subplot(2,4,4); imshow(reshape(s(:,2),size_ob(1),size_ob(2))); title('Output Pattern');
subplot(2,4,5); imshow(reshape(PN(:,3),size_ob(1),size_ob(2))); title('Input Pattern');
subplot(2,4,6); imshow(reshape(s(:,3),size_ob(1),size_ob(2))); title('Output Pattern');
subplot(2,4,7); imshow(reshape(PN(:,4),size_ob(1),size_ob(2))); title('Input Pattern');
subplot(2,4,8); imshow(reshape(s(:,4),size_ob(1),size_ob(2))); title('Output Pattern');

return 

function PN = makesomenoise(P) % This function was created for the purpose 
% of making noisy images given an  inital input matrix.

% Example : PN = makesomenoise(P) (PN is the noisy mtx) 

r = rand(size(P));  

PN = P; 

for i = 1:4 
    for j = 1:length(PN(:,i)) 
        if r(j,i) < .85 % Change about 75% of the matrix 
            PN(j,i) = 2*round(rand) -1;   
        end
    end 
end 

return 


function val = sign2(x) % Function containing the Hopefield threshold used
% used to update the state that inputs are currently in and used to obtain
% the output patterns.  
% Example:  ns = sign2(W*s) Used for obtaining new state calling the 
%threshold funciton 

tmp = sign(x); 
val = tmp + abs(tmp) - 1; 
return 

