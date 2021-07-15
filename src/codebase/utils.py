def make_dir_if_not_exist(d, remove=False):
    import os
    if remove and os.path.exists(d):  # USE AT YOUR OWN RISK
        import shutil
        shutil.rmtree(d)
    if not os.path.exists(d):
        os.makedirs(d)

def make_comma_separated_args_dirname_friendly(templates, overrides, transfer_overrides):
    if len(templates) > 0:
        string = " ".join(templates) + "," + ",".join(overrides)
    else:
        string = ",".join(overrides)
    if len(transfer_overrides) > 0:
        string += "/" + ",".join(transfer_overrides)
    dirname = string.replace("--", "").replace(" ", ",").replace(".", "_").replace("=", "-").replace(",", "--")
    return dirname


def get_npz_basename(name, biased, even, setnum=None, **kwargs):
    if name == 'german':
        if biased:
            basename = 'german_valid_inds_biased_valid_inds.npz'
        elif even:
            raise ValueError("even {} dataset not supported".format(name))
        else:
            basename = 'german_valid_inds.npz'
    elif name == 'compas':
        if biased:
            basename = 'compas_vr_valid_inds_biased_valid_inds.npz'
        elif even:
            raise ValueError("even {} dataset not supported".format(name))
        else:
            basename = 'compas_vr_valid_inds.npz'
    elif name == 'health':
        if biased:
            basename = 'health_Y2_charlson_biased_valid_inds.npz'
        elif even:
            basename = 'health_Y2_charlson_equal_valid_and_test.npz'
        else:
            basename = 'health_Y2_charlson.npz'
    elif name == 'adult':
        if biased:
            raise ValueError("biased {} dataset not supported".format(name))
        elif even:
           basename = 'adult_equal_valid_and_test.npz' 
        else:
            basename = 'adult.npz'
    elif name == 'diabetes':
        if biased:
            raise ValueError("biased {} dataset not supported".format(name))
        elif even:
            raise ValueError("even {} dataset not supported".format(name))
        else:
            basename = 'diabetes_edwards_clean.npz'
    elif name == 'adult_multi':
        basename = 'adult_multi.npz'
    elif name == 'run0':
        basename = 'run0_v2.npz'
    elif name == 'runhet':
        if setnum == '1':
            basename = 'run_agree_interact_common_20_061619_prod_1_4_try_1.npz'
        elif setnum == '2':
            basename = 'run_agree_interact_common_20_061619_prod_1_4_try_2.npz'
        elif setnum == '3':
            basename = 'run_agree_interact_common_20_061619_prod_1_4_try_3.npz'
        elif setnum == '4':
            basename = 'run_agree_interact_common_20_061619_prod_1_4_try_4.npz'
        elif setnum == '5':
            basename = 'run_agree_interact_common_20_061619_prod_1_4_try_5.npz'

        elif setnum == '6':
            basename = 'run_agree_interact_common_20_061619_prod_1_10_try_1.npz'
        elif setnum == '7':
            basename = 'run_agree_interact_common_20_061619_prod_1_10_try_2.npz'
        elif setnum == '8':
            basename = 'run_agree_interact_common_20_061619_prod_1_10_try_3.npz'
        elif setnum == '9':
            basename = 'run_agree_interact_common_20_061619_prod_1_10_try_4.npz'
        elif setnum == '10':
            basename = 'run_agree_interact_common_20_061619_prod_1_10_try_5.npz'

    elif name == 'runp1_2':
        basename = 'run_agree_no_interact_050919.npz'  #'run_p1_2_042219.npz'
    elif name == 'runp1_p5':
        basename = 'run_p1_p5.npz'
    elif name == 'runagree':
        basename ='run_agree_interact_common_20_061619_prod_2_10.npz'
    elif name == 'runorfunc':
        if setnum == '1':
            basename = 'run_orfunc_052619_4_try_1.npz'
        elif setnum == '2':
            basename = 'run_orfunc_052619_4_try_2.npz'
        elif setnum == '3':
            basename = 'run_orfunc_052619_4_try_3.npz'
        elif setnum == '4':
            basename = 'run_orfunc_052619_4_try_4.npz'
        elif setnum == '5':
            basename = 'run_orfunc_052619_4_try_5.npz'

        elif setnum == '6':
            basename = 'run_orfunc_052619_10_try_1.npz'
        elif setnum == '7':
            basename = 'run_orfunc_052619_10_try_2.npz'
        elif setnum == '8':
            basename = 'run_orfunc_052619_10_try_3.npz'
        elif setnum == '9':
            basename = 'run_orfunc_052619_10_try_4.npz'
        elif setnum == '10':
            basename = 'run_orfunc_052619_10_try_5.npz'
    elif name == 'bcesets':
        basename = 'bcesets_test_to_001_REAL.npz'
    elif name == 'shapes':
        basename = 'shapes_071719.npz'
    elif name == 'mnist_irm':
        if setnum == '1':
            basename = 'mnist_3study_digit100_unequal_color_try_1.npz'
        elif setnum == '2':
            basename = 'mnist_3study_digit100_unequal_color_try_2.npz'
        elif setnum == '3':
            basename = 'mnist_3study_digit100_unequal_color_try_3.npz'
        elif setnum == '4':
            basename = 'mnist_3study_digit100_unequal_color_try_4.npz'
        elif setnum == '5':
            basename = 'mnist_3study_digit100_unequal_color_try_5.npz'

        elif setnum == '6':
            basename = 'mnist_6study_digit100_unequal_color_try_1.npz'
        elif setnum == '7':
            basename = 'mnist_6study_digit100_unequal_color_try_2.npz'
        elif setnum == '8':
            basename = 'mnist_6study_digit100_unequal_color_try_3.npz'
        elif setnum == '9':
            basename = 'mnist_6study_digit100_unequal_color_try_4.npz'
        elif setnum == '10':
            basename = 'mnist_6study_digit100_unequal_color_try_5.npz'
    elif name == 'mnist':
        if setnum == '1':
            basename = 'mnist_digit75_color80test_5050_041020_try_1.npz'
        elif setnum == '2':
            basename = 'mnist_digit75_color80test_5050_041020_try_2.npz'
        elif setnum == '3':
            basename = 'mnist_digit75_color80test_5050_041020_try_3.npz'
        elif setnum == '4':
            basename = 'mnist_digit75_color80test_5050_041020_try_4.npz'
        elif setnum == '5':
            basename = 'mnist_digit75_color80test_5050_041020_try_5.npz'

        elif setnum == '6':
            basename = 'mnist_digit100_color90flipped_testpurple_022120_try_1.npz'
        elif setnum == '7':
            basename = 'mnist_digit100_color90flipped_testpurple_022120_try_2.npz'
        elif setnum == '8':
            basename = 'mnist_digit100_color90flipped_testpurple_022120_try_3.npz'
        elif setnum == '9':
            basename = 'mnist_digit100_color90flipped_testpurple_022120_try_4.npz'
        elif setnum == '10':
            basename = 'mnist_digit100_color90flipped_testpurple_022120_try_5.npz'
    elif name == 'mnist_simple':
        basename = 'mnist_digit100_color90flipped_testpurple_022120.npz'
        # if setnum == '1':
        #     basename = 'mnist_4060_4060_unbalanced_digit75_color80flipped_test_6040_032620_set_1.npz'
        # elif setnum == '2':
        #     basename = 'mnist_4060_4060_unbalanced_digit75_color80flipped_test_6040_032620_set_2.npz'
        # elif setnum == '3':
        #     basename = 'mnist_4060_4060_unbalanced_digit75_color80flipped_test_6040_032620_set_3.npz'
        # elif setnum == '4':
        #     basename = 'mnist_4060_4060_unbalanced_digit75_color80flipped_test_6040_032620_set_4.npz'
        # elif setnum == '5':
        #     basename = 'mnist_4060_4060_unbalanced_digit75_color80flipped_test_6040_032620_set_5.npz'
        # else:
        #     basename = setnum + " ERROR"
    elif name == 'mnist_3':
        basename = 'mnist_digit100_study1red_study2green_testblue_041520.npz'
    elif name == 'pacs':
        if setnum == '1':
            basename = 'pacs_P_test_042420_64.npz'
        elif setnum == '2':
            basename = 'pacs_A_test_042420_64.npz'
        elif setnum == '3':
            basename = 'pacs_C_test_042420_64.npz'
        elif setnum == '4':
            basename = 'pacs_S_test_042420_64.npz'
        
    return basename
