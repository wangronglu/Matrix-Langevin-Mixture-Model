function y = log_langevin_pdf(X,F,tao)
%%% compute the pdf of X(:,:,i) when X(:,:,i) ~ matrix Langevin (F)
%%% X is a VxKxn
%%% X(:,:,i) is a VxK orthonormal matrix for each i
%%% F is a VxK matrix
%%% tao is the vector of singular values of F
%%% return a nx1 vector y

V = size(X,1);
n = size(X,3);
%%% log f(X|F) = tr(F'X) - log 0_F_1(V/2;tao'tao/4)
% log_norm_cons = logmhg(50,2,[],[V/2],[tao.*tao/4]); % a crude estimate,  truncated for partitions of size not exceeding 20
% log_norm_cons = log0F1_2(V,tao);
log_norm_cons = log0F1(V,tao);

y = zeros(n,1);
for i=1:n
    y(i) = sum(sum(F.*X(:,:,i))) ;
end
y = y - log_norm_cons;
