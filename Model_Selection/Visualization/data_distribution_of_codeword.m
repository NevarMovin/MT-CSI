clear
clc
load('codeword_out_CR16.mat');
sample_CR16_out = codeword_outdoor(1:200,:);
size(sample_CR16_out)
mesh(sample_CR16_out)
