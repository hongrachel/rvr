# Representations via Representations
Frances Ding, Prasad Patil, Giovanni Parmigiani, Cynthia Dwork

Representation via Representations is a project aimed at improving transfer learning to out-of-distribution examples. Motivated by the challenge of finding robust biomedical predictors of disease, the model leverages data from heterogenous sources to discover feature representations that allow for accurate prediction outside of the training data.

This codebase owes its foundations to David Madras, Elliot Creager, Toni Pitassi, Richard Zemel in their paper Learning Adversarially Fair and Transferable Representations (https://arxiv.org/abs/1802.06309). Github link: https://github.com/VectorInstitute/laftr

## setting up a project-specific virtual env
```
mkdir ~/venv 
python3 -m venv ~/venv/rvr
```
where `python3` points to python 3.6.X. Then
```
source ~/venv/rvr/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
or 
```
pip install -r requirements-gpu.txt
```
for GPU support

## running a single fair classification experiment
```
source simple_example.sh
```
The bash script first trains LAFTR and then evaluates by training a naive classifier on the LAFTR representations (encoder outputs).
See the paper for further details.

## running a sweep of fair classification with various hyperparameter values
```
python src/generate_sweep.py sweeps/small_sweep_adult/sweep.json
source sweeps/small_sweep_adult/command.sh
```
The above script is a small sweep which only trains for a few epochs. 
It is basically just for making sure everything runs smoothly. 
For a bigger sweep call `src/generate_sweep` with `sweeps/full_sweep_adult/sweep.json`, or design your own sweep config.

## data
The (post-processed) adult dataset is provided in `data/adult/adult.npz`

