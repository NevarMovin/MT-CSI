clear
clc
load('codeword_in_CR4.mat');
sample_CR4_in = codeword_indoor(1,:);
sample_CR4_in_2d = reshape(sample_CR4_in, 32, 16);
imagesc(sample_CR4_in_2d)
