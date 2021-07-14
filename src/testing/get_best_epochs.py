import numpy as np
import itertools
import os
import sys
import argparse
sys.path.append("/Users/rachelh/Programs/rvr/src/")
sys.path.append("~/rvr/src/")
sys.path.append("/n/home01/rhong/rvr/src/testing/")
from find_test_result import get_ckpt_stats, loss, loss_subset

BIG = 99999.

#def num2str(x):
#    if np.isclose(x % 1, 0.):
#        return '{:d}'.format(int(x))
#    else:
#        return '{:.1f}'.format(x)

def num2str(x, digits=None):
    if digits is None:
        if np.isclose(x % 1, 0.):
            return '{:d}'.format(int(x))
        else:
            i = 1
            while not np.isclose(x * (10 ** i) % 1, 0.) and i < 8:
                i += 1
            format_str = '{:.' + '{:d}'.format(i) + 'f}'
            return format_str.format(x)
    else:
        format_str = '{:.' + '{:d}'.format(digits) + 'f}'
        return format_str.format(x)

def get_best_epoch(expdir, fairkey, lamda):
    bigdir = os.path.join(expdir, 'checkpoints')

    # loop through all validation directories
    ckpt_names = os.listdir(bigdir)
    valid_ckpt_names = filter(lambda s: 'Valid' in s, ckpt_names)

    best_loss = BIG
    best_epoch = -1
    best_err = BIG
    best_dp = BIG
    # print(fairkey)
    for d in valid_ckpt_names:
        dname = os.path.join(bigdir, d)

        stats = get_ckpt_stats(dname)
        l, err, fair, _, _, _ = loss(stats, lamda, fairkey, unsup=False)
        if l < best_loss:
            ep = int(d.split('_')[1])
            # print(ep, l, err, fair)
            best_loss, best_err, best_fair = l, err, fair
            best_epoch = ep

    return best_epoch, best_fair

def get_best_epoch_erry(expdir):
    bigdir= os.path.join(expdir, 'checkpoints')

    ckpt_names = os.listdir(bigdir)
    valid_ckpt_names = filter(lambda s: 'Valid' in s, ckpt_names)

    best_err = BIG
    best_epoch = -1

    for d in valid_ckpt_names:
        dname = os.path.join(bigdir, d)
        stats = get_ckpt_stats(dname)
        err, classce, discce = loss_subset(stats)
        if err < best_err:
            ep = int(d.split('_')[1])
            best_err = err
            best_epoch = ep

    return best_epoch, best_err

def write_best_epochs(fname, best_eps, lamda):
    f = open(fname, 'w')
    f.write('Lamda,{:.3f}\n'.format(lamda))
    for fm in sorted(best_eps):
        if np.isnan(best_eps[fm][0]): print(fname, fm, 'is nan')
        # print('{},{:d}\n'.format(fm, best_eps[fm]))
        f.write('{},{:d},{:.5f}\n'.format(fm, best_eps[fm][0], best_eps[fm][1]))
    f.close()

if __name__ == '__main__':

    # Original from DM, EC:
    '''
    if revert_to_original:
        # expdirs = ['/ais/gobi5/madras/adv-fair-reps/experiments/health_eqopp_whw_fc6_ua_2']
        expdir = '/ais/gobi5/madras/adv-fair-reps/experiments/'
        dp_dirs = ['adult_dempar_whw_fc{}'.format(num2str(gamma)) \
                   for gamma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]] + \
                 ['health_dempar_whw_fc{}_ua'.format(num2str(gamma)) \
                  for gamma in [1, 1.5, 2, 2.5, 3, 3.5, 4]]
        eqodds_dirs = ['adult_eqodds_whw_fc{}_2'.format(num2str(gamma)) \
                   for gamma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]] + \
                        ['health_eqodds_whw_fc{}_ua_2'.format(num2str(gamma)) \
                         for gamma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]]
        eqopp_dirs = ['adult_eqopp_whw_fc{}_2'.format(num2str(gamma)) \
                   for gamma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]] + \
                        ['health_eqopp_whw_fc{}_ua_2'.format(num2str(gamma)) \
                         for gamma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]]
        expdirs = [os.path.join(expdir, d) for d in dp_dirs + eqodds_dirs + eqopp_dirs]
        fmets = ['DP', 'DI', 'DI_FP', 'ErrA']
        lamda = 1.
        
        # somehow loop over all runs
        for d in expdirs:
            # for each fairness metric
            best_eps = {}
            for fm in fmets:
                best_ep, best_fair = get_best_epoch(d, fm, lamda)
                best_eps[fm] = (best_ep, best_fair)
            fname = os.path.join(d, 'best_validation_fairness.txt')
            write_best_epochs(fname, best_eps, lamda)
            print('Wrote metrics to {}'.format(fname))
        
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='name of experiment')
    parser.add_argument('--seed', type=int, help='random seed value')
    parser.add_argument('--adim', type=int, help='attribute vector dimension')
    args=parser.parse_args()

    on_laptop = False
    if on_laptop:
        expdir = '/Users/Frances/Documents/seas-fellowship/rvr/experiments/'
    else:
        expdir = '/n/scratchlfs/macklis_lab/fding/rvr/experiments' 
        #expdir = '/n/home06/fding/rvr/experiments/'


    #run0_dirs = ['run0_sweep/data--run0--model_adim-10--model_class-WeightedDemParMultiWassGan--model_fair_coeff-{}'.format(gamma) \
    #             for gamma in ['0_0', '0_5', '1_0', '2_0', '4_0', '6_0', '10_0', '20_0', '40_0']]
    #runhet_dir = 'runhet_sweep/data--runhet--model_adim-10--model_class-WeightedDemParMultiWassGan--model_fair_coeff-{}'
    #runhet_dirs = [runhet_dir.format(gamma) for gamma in \
    #               ['0_0', '0_005', '0_01', '0_05', '0_1', '0_2', '0_5', '1_0', '2_0', '4_0', '6_0', '8_0', '10_0', '20_0', '50_0', '100_0']]
    #expdirs = [os.path.join(expdir, d) for d in runhet_dirs]

    #run0_v2_dirs = ['run0_sweep_v2_no_attr/data--run0--model_adim-10--model_class-WeightedDemParMultiWassGan--model_fair_coeff-{}'.format(gamma) \
    #             for gamma in ['0_0', '0_5', '1_0', '2_0', '4_0', '6_0', '8_0', '10_0', '20_0', '50_0', '100_0']]

    #expdirs = [os.path.join(expdir, d) for d in run0_v2_dirs]


    # Runhet_recon sweeps
    '''
    coeffs = ['0_0', '0_005', '0_01', '0_05', '0_1', '0_2', '0_5', '1_0', '2_0', '4_0', '6_0', '10_0']

    runhet_dir = 'runhet_sweep/data--runhet--model_adim-10--model_class-WeightedDemParMultiWassGan--model_fair_coeff-{}'

    runhet_recon_dir = 'runhet_recon_sweep/data--runhet--model_adim-10--model_class-WeightedDemParMultiWassGan--model_fair_coeff-{}--model_recon_coeff-{}'
    runhet_recon_dirs = [(runhet_recon_dir.format(gamma, beta), gamma, beta) for gamma, beta in itertools.product(coeffs, coeffs)]
    '''

    #coeffs = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 4.0]

    #fair_coeffs = ['0_0', '0_01', '0_05', '0_1', '0_15', '0_2', '0_3', '0_5', '1_0', '4_0']
    #recon_coeffs = ['0_0', '0_001', '0_005', '0_01', '0_03', '0_05', '0_1', '0_15', '0_2', '0_3']

    #fair_coeffs = ['0_0', '0_1', '0_5', '1_0', '3_0', '5_0', '10_0', '15_0']
    #recon_coeffs = ['0_0', '0_0005', '0_001', '0_002', '0_003', '0_005', '0_007', '0_01', '0_03', '0_05', '0_1'] #'0_7'

    #fair_coeffs = ['0_0', '0_5', '1_0', '3_0', '5_0', '10_0', '15_0']
    #recon_coeffs = ['0_0', '0_002', '0_005', '0_01', '0_03', '0_05', '0_07', '0_1']

    #fair_coeffs = ['0_0', '0_1', '0_5', '1_0', '3_0', '5_0', '10_0', '15_0']
    #recon_coeffs = ['0_0', '0_001', '0_005', '0_01', '0_025', '0_05', '0_075', '0_1', '0_3', '0_5']

    #fair_coeffs = ['0_0', '1_0', '3_0', '5_0', '10_0', '15_0']
    #recon_coeffs = ['0_0', '0_005', '0_01', '0_03', '0_05', '0_07', '0_1']

    fair_coeffs = ['0_0', '1_0', '3_0', '5_0', '10_0', '15_0']
    recon_coeffs = ['0_0', '0_005']

    #sweepname = 'runp1_2_sweep_eo_042219'
    #seed = args.seed
    #runp_dir = '{}/data--runp1_2--model_adim-10--model_class-WeightedDemParMultiWassGan--model_fair_coeff-{}--model_recon_coeff-{}'
    #runp_dir = '{}/data--runp1_2--model_adim-4--model_class-MultiEqOddsUnweightedWassGan--model_fair_coeff-{}--model_recon_coeff-{}--model_seed-{}'
    #runp_dirs = [(runp_dir.format(sweepname, gamma, beta, seed), gamma, beta) for gamma, beta in itertools.product(fair_coeffs, recon_coeffs)]


    sweepname = args.exp #'runagree_all6060_interact_052619_prod_10'
    seed = args.seed
    adim = args.adim
    runagree_dir = '{}/data--runagree--model_adim-{}--model_class-MultiEqOddsUnweightedWassGan--model_fair_coeff-{}--model_recon_coeff-{}--model_seed-{}'
    runagree_dirs = [(runagree_dir.format(sweepname, adim, gamma, beta, seed), gamma, beta) for gamma, beta in itertools.product(fair_coeffs, recon_coeffs)]

    orfunc = False
    if orfunc:
        sweepname = args.exp #'runorfunc_all6060_052619_10'
        adim = args.adim
        seed = args.seed
        runagree_dir = '{}/data--runorfunc--model_adim-{}--model_class-MultiEqOddsUnweightedWassGan--model_fair_coeff-{}--model_recon_coeff-{}--model_seed-{}'
        runagree_dirs = [(runagree_dir.format(sweepname, adim, gamma, beta, seed), gamma, beta) for gamma, beta in itertools.product(fair_coeffs, recon_coeffs)]

    bcesets = True
    if bcesets:
        sweepname = args.exp #'runorfunc_all6060_052619_10'
        adim = args.adim
        seed = args.seed
        runagree_dir = '{}/data--bcesets--model_adim-{}--model_class-MultiEqOddsUnweightedWassGan--model_fair_coeff-{}--model_recon_coeff-{}--model_seed-{}'
        runagree_dirs = [(runagree_dir.format(sweepname, adim, gamma, beta, seed), gamma, beta) for gamma, beta in itertools.product(fair_coeffs, recon_coeffs)]

    #expdirs = [(os.path.join(expdir, d), gamma, beta) for d, gamma, beta in runp_dirs]
    expdirs = [(os.path.join(expdir, d), gamma, beta) for d, gamma, beta in runagree_dirs]

    score_mat = []
    valid_score_mat = []

    for d, gamma, beta in expdirs:
        best_epoch, best_err = get_best_epoch_erry(d)
        #print(d)
        #print(best_epoch, best_err)
        testdir = d + '/checkpoints/Epoch_{}_Test/'.format(best_epoch)
        teststats = get_ckpt_stats(testdir)
        err, classce, discce = loss_subset(teststats)
        #print(d[-5:])
        #print('Gamma: {} Beta: {}'.format(gamma, beta))
        #print(err, classce, discce)
        score_mat.append(err)
        valid_score_mat.append(best_err)

    score_mat = np.reshape(np.array(score_mat), (len(fair_coeffs), -1))
    valid_score_mat = np.reshape(np.array(valid_score_mat), (len(fair_coeffs), -1))

    #print(score_mat)

    np.save('{}_{}_score_mat.npy'.format(sweepname, seed), score_mat)
    np.save('{}_{}_valid_score_mat.npy'.format(sweepname, seed), valid_score_mat)

    save_csv = False
    if save_csv:
        import csv

        with open('{}_results.csv'.format(sweepname), mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in score_mat:
                writer.writerow(row)



    # get reprs from best epoch

    # run accuracy-only mlp

    # record the fairness metrics


    #TODO: Re-run Adult with sensitive attribute
    #TODO: Re-run transfer stuff with whw
    #TODO: Try comparing xent with whw on transfer????
