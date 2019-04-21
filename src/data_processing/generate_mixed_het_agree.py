# generate_mixed_het_agree.py
#
# Generates data with mixed heterogeneity with agreement between the common and uncommon factors about the label
#
''' Parameters to set:

# K: number of studies
# nk: sample size of each study
# p: number of covariates
# p_c: number of common covariates (c <= p)
# mu - K*p matrix of covariate means
# SIG - p*p covariance matrix
# eps - perturbation level for homogenous covariates
# eta - perturbation level for heterogeneous covariates
# beta_min - minimum for beta window
# beta_max - maximum for beta window
'''

# Beta are generated from [-beta_max, -beta_min] U [beta_min, beta_max]
# for beta_k in p_c, the study-specific beta_k is taken from [beta_k - eps, beta_k + eps]
# for beta_k not in p_c, the study-spceific beta_k is taken from [beta_k - eta, beta_k + eta]

import numpy as np

def perturb_betas(beta_vec, k, c_idx, eps, eta):
    out = np.zeros((k, beta_vec.shape[0]))
    for i in range(beta_vec.shape[0]): # for coefficient i
        if i in c_idx:
            out[:, i] = np.random.uniform(beta_vec[i] - eps, beta_vec[i] + eps, size=k)
        else:
            out[:, i] = np.random.uniform(beta_vec[i] - eta, beta_vec[i] + eta, size=k)
    return out

def compute_label(x_vec, beta_vec, exact=False):
    if exact:
        if x_vec @ beta_vec > 0:
            return 1
        else:
            return 0
    else:
        z = 1.0 / (1 + np.exp(-(x_vec @ beta_vec)))
        out = np.random.binomial(1, z, len(z))
    return out

def multi_study_sim(k, nk, p, p_c, mu, sig, eps, eta, beta_min, beta_max, outfile):

    # generate common features
    c_idx = np.random.choice(range(p), size=p_c, replace=False) # indices of common features
    non_c_idx = list(set(range(p)).difference(c_idx))
    #mask = [0 if x in c_idx else 1 for x in range(p)]
    #notmask = [1-x for x in mask]


    # generate 'true' betas
    beta_vec = np.random.uniform(beta_min, beta_max, size=p)
    beta_vec = np.array([x if np.random.rand() < 0.5 else -x for x in beta_vec ]) #make about half negative

    # generate study-specific betas
    beta_vec_list = perturb_betas(beta_vec, k, c_idx, eps, eta)

    # generate each study's covariates with a form of rejection sampling
    baserates = np.random.uniform(0.3, 0.7, size=k)

    x_train = None
    x_test = None
    y_train = None
    y_test = None

    for i in range(k):
        numpos = int(np.around(baserates[i] * nk[i])) # total number of positives we want in the end
        numneg = int(nk[i]) - numpos

        # construct the positives
        poscount = 0
        totalcount = 0
        x_vec_pos = np.zeros((numpos, p))
        while poscount < numpos:
            x_vec = np.random.multivariate_normal(mean=mu[i], cov=sig, size=1)

            # find label from common features
            y = compute_label(x_vec=x_vec[:,c_idx], beta_vec=beta_vec_list[i, c_idx])

            # check if uncommon features agree
            y_uncommon = compute_label(x_vec=x_vec[:,non_c_idx], beta_vec=beta_vec_list[i, non_c_idx], exact=True)
            totalcount += 1

            if (y == 1 and y_uncommon == 1): # if both agree on positive, add to vector
                x_vec_pos[poscount, :] = x_vec
                poscount += 1

        # construct the negatives
        negcount = 0
        totalcount = 0
        x_vec_neg = np.zeros((numneg, p))
        while negcount < numneg:
            x_vec = np.random.multivariate_normal(mean=mu[i], cov=sig, size=1)

            # find label from common features
            y = compute_label(x_vec=x_vec[:, c_idx], beta_vec=beta_vec_list[i, c_idx])

            # check if uncommon features agree
            y_uncommon = compute_label(x_vec=x_vec[:, non_c_idx], beta_vec=beta_vec_list[i, non_c_idx], exact=True)
            totalcount += 1

            if (y == 0 and y_uncommon == 0): # if both agree on positive, add to vector
                x_vec_neg[negcount, :] = x_vec
                negcount += 1

        x_vec_out = np.concatenate((x_vec_pos, x_vec_neg), axis=0)
        y_vec = np.concatenate((np.ones(numpos), np.zeros(numneg)))

        if i == 0: # if first study, save as final file
            x_train = x_vec_out
            y_train = y_vec
        elif i < k - 1: # if further training studies, concatenate
            x_train = np.concatenate((x_train, x_vec_out), axis=0)
            y_train = np.concatenate((y_train, y_vec), axis=0)
        else: # last study is test study
            x_test = x_vec_out
            y_test = y_vec
        """x_vec = np.random.multivariate_normal(mean=mu[i], cov=sig, size=int(4*nk[i])) # 4x as many datapoints as needed
        y_vec = compute_label(x_vec[:, c_idx], beta_vec_list[i, c_idx]) # labels based on common features
        print(beta_vec_list[i])
        print(y_vec)
        pos_idx = np.where(y_vec == 1)[0] # indices where y is positive
        neg_idx = np.where(y_vec == 0)[0]
        assert(len(pos_idx) > numpos)
        assert(len(neg_idx) > numneg)

        final_pos_idx = pos_idx[:numpos]
        final_neg_idx = neg_idx[:numneg]

        x_vec_pos = x_vec[final_pos_idx, :]
        x_vec_neg = x_vec[final_neg_idx, :]

        #print(x_vec_pos[:, c_idx] @ beta_vec_list[i, c_idx])
        #print(x_vec_neg[:, c_idx] @ beta_vec_list[i, c_idx])



        x_vec_out = np.concatenate((x_vec_pos, x_vec_neg), axis = 0)
        print(x_vec_out)"""


    # make y vectors 2-D
    y_train = np.expand_dims(y_train, 1)
    y_train_expand = np.concatenate((1 - y_train, y_train), axis=1)
    y_test = np.expand_dims(y_test, 1)
    y_test_expand = np.concatenate((1 - y_test, y_test), axis=1)

    # create attr matrices
    ind = np.repeat(0, nk[0])
    for i in range(1, k-1):
        ind = np.concatenate((ind, np.repeat(i, nk[i])), axis=0)
    attr_train = np.zeros((ind.size, ind.max()+1))
    attr_train[np.arange(ind.size), ind] = 1
    attr_test = np.concatenate( (np.ones((int(nk[-1]), 1)), np.zeros((int(nk[-1]), k-2))), axis=1)

    # create train and valid inds
    numtrainidx = int(0.8 * x_train.shape[0])
    shuffled = np.random.permutation(np.arange(x_train.shape[0]))
    train_inds = shuffled[:numtrainidx]
    valid_inds = shuffled[numtrainidx:]

    # save outfile
    np.savez(outfile, x_train=x_train, x_test=x_test, y_train=y_train_expand, y_test=y_test_expand,
             attr_train=attr_train, attr_test=attr_test, train_inds=train_inds, valid_inds=valid_inds,
             c_idx=c_idx, beta_vec_list=beta_vec_list)




if __name__ == '__main__':
    # Save file name:
    outfile = 'run_agree_p1_2_042019'

    # Set parameters for run
    np.random.seed(0)
    K = 5 # Total number of studies
    K_train = K-1 # number of training studies
    nk = np.ones(K)*5000 #5000 # number of observations per study, currently all same
    p = 30 # number of covariates
    p_c = 7 # number of common covariates
    eps = 0.1 # window size for common covariates
    eta = 2 # window size for non-comman covariates
    beta_min = 0.25 # beta window minimum
    beta_max = 2 # beta window maximum

    # covariate means
    mu = np.random.uniform(-3, 3, size=K*p).reshape((K, p))

    # SIG diagonal
    #sig = np.identity(p)

    # SIG arbitrary
    arb = np.random.uniform(-1, 1, size=p*p).reshape((p, p))
    sig = arb.T @ arb

    multi_study_sim(k=K, nk=nk, p=p, p_c=p_c, mu=mu, sig=sig, eps=eps, eta=eta, beta_min=beta_min, beta_max=beta_max,
                    outfile=outfile)
