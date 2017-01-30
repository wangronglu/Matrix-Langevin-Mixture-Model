function [R, llh] = expectation(X, model)
F_array = model.F;
tao_array = model.tao;
phi = model.phi;

n = size(X,3);
m = size(F_array,3);
R = zeros(n,m);

for j=1:m
    R(:,j) = log_langevin_pdf(X, F_array(:,:,j), tao_array(:,j));
end

R = bsxfun(@plus,R,log(phi));
T = logsumexp(R,2);
llh = sum(T); % joint loglikelihood
R = exp(bsxfun(@minus,R,T)); % conditional probability that data i arises from the j-th component
