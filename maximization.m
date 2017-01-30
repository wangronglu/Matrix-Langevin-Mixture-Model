function model = maximization(X, R)
%%% the M-step in EM algorithm
%%% X is VxKxn array storing the VxK orthogonal matrix for each subject
%%% R is weight matrix storing the posterior weights for each subject

[V,K,n] = size(X);
m = size(R,2); % number of clusters
cluster_R = sum(R,1); % for each cluster accumulate weights over all the subjects
phi = cluster_R/sum(cluster_R); % mixing weight vector in the model

%% compute the weighted average matrix in each cluster
norm_clust_R = exp(bsxfun(@minus,log(R),log(cluster_R) )); % normalized weight within each cluster
X_vec = reshape(X,V*K,n);
X_bar = X_vec * norm_clust_R;
X_bar = reshape(X_bar,[V K m]);

%% compute the MLE for Langevin distribution
% svd: F = G * tao * H' 
F_hat = zeros(V,K,m);
tao_hat = zeros(K,m);
options = optimoptions('fsolve','Display','none');
for j=1:m
   [G_hat,Lam,H_hat] = svd( X_bar(:,:,j), 'econ' ); 
   % fun = @(x) root5d(x,diag(Lam) );
   fun = @(x) root4d(x,diag(Lam),V,K);
   tao_hat(:,j) = fsolve(fun, V*diag(Lam),options); % initial value set at V*diag(Lam)
   F_hat(:,:,j) = G_hat * diag(tao_hat(:,j)) * H_hat';
end
%%
model.phi = phi;
model.F = F_hat;
model.tao = tao_hat;