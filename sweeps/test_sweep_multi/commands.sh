python src/run_laftr.py sweeps/test_sweep_multi/config.json --dirs local --data adult_multi -o model.adim=5,model.class=WeightedDemParMultiWassGan,model.fair_coeff=0.0,exp_name="test_sweep_multi/data--adult_multi--model_adim-5--model_class-WeightedDemParMultiWassGan--model_fair_coeff-0_0"
python src/run_laftr.py sweeps/test_sweep_multi/config.json --dirs local --data adult_multi -o model.adim=5,model.class=WeightedDemParMultiWassGan,model.fair_coeff=1.0,exp_name="test_sweep_multi/data--adult_multi--model_adim-5--model_class-WeightedDemParMultiWassGan--model_fair_coeff-1_0"
python src/run_laftr.py sweeps/test_sweep_multi/config.json --dirs local --data adult_multi -o model.adim=5,model.class=WeightedDemParMultiWassGan,model.fair_coeff=4.0,exp_name="test_sweep_multi/data--adult_multi--model_adim-5--model_class-WeightedDemParMultiWassGan--model_fair_coeff-4_0"
