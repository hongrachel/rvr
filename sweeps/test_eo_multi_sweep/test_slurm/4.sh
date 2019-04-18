SBATCH something something
SBATCH something else

python src/run_laftr.py sweeps/test_eo_multi_sweep/config.json --dirs local --data runp1_2 -o model.adim=10,model.class=MultiEqOddsUnweightedWassGan,model.fair_coeff=0.0,model.recon_coeff=0.05,exp_name="test_eo_multi_sweep/data--runp1_2--model_adim-10--model_class-MultiEqOddsUnweightedWassGan--model_fair_coeff-0_0--model_recon_coeff-0_05"
