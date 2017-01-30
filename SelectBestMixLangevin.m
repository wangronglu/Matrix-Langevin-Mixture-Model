function model_best = SelectBestMixLangevin(V,K,EigenVec_array,upperg,uppiter,maxiter,rep_num,method_ind)
% warning! dichotomy may only select a local optimal and miss the global optimal. 
% This function uses dichotomy algorithm to select the optimal number of
% components in mixture Langevin distribution over VxK Stiefel manifold
% EigenVec_array is VxKxn data
% upperg is the upper bound of number of components
% uppiter is the upper bound of iterations in dichotomy algorithm
% maxiter is the upper bound of iterations in EM algorithm for likelihood
% to converge
% rep_num is number of repetitions in K-means initialization
% method_ind is indicator of 'BIC' or 'ICL'

if (nargin < 8)
    error('Less number of inputs');
end

n=size(EigenVec_array,3);

if (strcmp(method_ind,'BIC')==1)
    m1=1;
    [model_BIC_m1,model_ICL_m1] = EM_Mix_Langevin(V,K,n,m1,EigenVec_array,maxiter,rep_num);
    model_BIC_best = model_BIC_m1;
    
    m2 = upperg;
    [model_BIC_m2,model_ICL_m2] = EM_Mix_Langevin(V,K,n,m2,EigenVec_array,maxiter,rep_num);
    if model_BIC_m2.llh > model_BIC_m1.llh
        model_BIC_best = model_BIC_m2;
    end
    
    for s=2:uppiter
        if (m2 - m1 == 1)
            disp(m1);
            break;
        end
        m = floor( (m1+m2)/2 );
        [model_BIC_m,model_ICL_m] = EM_Mix_Langevin(V,K,n,m,EigenVec_array,maxiter,rep_num);
        if (model_BIC_m.llh > model_BIC_m2.llh)
            m2 = m;
            model_BIC_m2 = model_BIC_m;
      
        elseif (model_BIC_m.llh > model_BIC_m1.llh)
            m1 = m;
            model_BIC_m1 = model_BIC_m;
        end
        
        if (model_BIC_m2.llh > model_BIC_m1.llh)
            model_BIC_best = model_BIC_m2;
        else
            model_BIC_best = model_BIC_m1;
        end
        disp(m);
    end
    model_best = model_BIC_best;
elseif (strcmp(method_ind,'ICL')==1)
    m1=1;
    [model_BIC_m1,model_ICL_m1] = EM_Mix_Langevin(V,K,n,m1,EigenVec_array,maxiter,rep_num);
    model_ICL_best = model_ICL_m1;
    
    m2 = upperg;
    [model_BIC_m2,model_ICL_m2] = EM_Mix_Langevin(V,K,n,m2,EigenVec_array,maxiter,rep_num);
    if model_ICL_m2.llh > model_ICL_m1.llh
        model_ICL_best = model_ICL_m2;
    end
    
    for s=2:uppiter
        if (m2 - m1 == 1)
            disp(m1);
            break;
        end
        m = floor( (m1+m2)/2 );
        [model_BIC_m,model_ICL_m] = EM_Mix_Langevin(V,K,n,m,EigenVec_array,maxiter,rep_num);
        if (model_ICL_m.llh > model_ICL_m2.llh)
            m2 = m;
            model_ICL_m2 = model_ICL_m;
            
        elseif (model_ICL_m.llh > model_ICL_m1.llh)
            m1 = m;
            model_ICL_m1 = model_ICL_m;
        end
        
        if (model_ICL_m2.llh > model_ICL_m1.llh)
            model_ICL_best = model_ICL_m2;
        else
            model_ICL_best = model_ICL_m1;
        end
        disp(m);
    end
    model_best = model_ICL_best;
else
    error('method_ind should be BIC or ICL')
end