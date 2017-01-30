function [model_BIC_m,model_ICL_m] = EM_Mix_Langevin(V,K,n,m,EigenVec_array,maxiter,rep_num)
% This function estimate mixture Langevin distribution on VxK Stiefel
% manifold (orthonormal matrices) given the number of components 
% n is the size of the data
% m is the number of components in mixture model
% EigenVec_array contains the VxKxn orthonormal matrices
% maxiter is the maximum iterations for likelihood to converge in EM
% rep_num is the number of repetitions in K-means initialization

if (m == 1)
    label = ones(1,n);
    R = full(sparse(1:n,label,1,n,m,n)); % n x m cluster indicator matrix/posterior weight matrix
    tol = 1e-6;
    % maxiter = 30;
    llh = -inf(maxiter,1);
    
    for iter = 2:maxiter
        label = find(sum(R,1)); % return index of non-empty cluster
        R = R(:,label);
        model = maximization(EigenVec_array,R);
        [R,llh(iter)] = expectation(EigenVec_array, model);
        model.R = R;
        if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter));
            disp(iter);
            break;
        end;
        % disp(iter);
    end
    model_BIC_m = model;
    model_ICL_m = model;
    
    model_BIC_m.llh = llh(iter) - log(n)*(m*V*K+m-1)/2;
    model_ICL_m.llh = llh(iter) - log(n)*(m*V*K+m-1)/2;
    return;
    
elseif (m > 1)
    eigendata = reshape(EigenVec_array,V*K,n);
    eigen_dist = pdist(eigendata');
    eigen_hclust = linkage(eigen_dist,'complete');
    label = cluster(eigen_hclust,'maxclust',m);
    label = label';
    
    R = full(sparse(1:n,label,1,n,m,n)); % n x m cluster indicator matrix/posterior weight matrix
    tol = 1e-6;
    % maxiter = 50;
    llh = -inf(maxiter,1);
    
    for iter = 2:maxiter
        label = find(sum(R,1)); % return index of non-empty cluster
        R = R(:,label);
        model = maximization(EigenVec_array,R);
        [R,llh(iter)] = expectation(EigenVec_array, model);
        model.R = R;
        if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter));
            disp(iter);
            break;
        end;
        % disp(iter);
    end
    
    BIC_m = llh(iter) - log(n)*(m*V*K+m-1)/2;
    model_BIC_m = model; % best model under m
    
    ICL_m = llh(iter) + sum(log(max(model.R,[],2))) - log(n)*(m*V*K+m-1)/2; % integrated complete likelihood
    model_ICL_m = model;
    
    % random initialization: label = floor(m*rand(1,n))+1;
    %% initializing with K-means
    % rep_num = 5;
    
    for kiter = 1:rep_num
        label = kmeans(eigendata',m);
        label = label';
        
        R = full(sparse(1:n,label,1,n,m,n)); % n x m cluster indicator matrix/posterior weight matrix
        tol = 1e-6;
        % maxiter = 50;
        llh = -inf(maxiter,1);
        
        for iter = 2:maxiter
            label = find(sum(R,1)); % return index of non-empty cluster
            R = R(:,label);
            model = maximization(EigenVec_array,R);
            [R,llh(iter)] = expectation(EigenVec_array, model);
            model.R = R;
            if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter));
                disp(iter);
                break;
            end;
            % disp(iter);
        end
        
        BIC = llh(iter) - log(n)*(m*V*K+m-1)/2;
        if BIC > BIC_m
            BIC_m = BIC;
            model_BIC_m = model;
        end
        
        ICL = llh(iter) + sum(log(max(model.R,[],2))) - log(n)*(m*V*K+m-1)/2; % integrated complete likelihood
        if ICL > ICL_m
            ICL_m = ICL;
            model_ICL_m = model;
        end
        % disp(kiter);
    end
    
    model_BIC_m.llh = BIC_m;
    model_ICL_m.llh = ICL_m;
    return;
    
else
    error('The number of components m should be a positive integer.')
end