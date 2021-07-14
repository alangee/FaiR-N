# FaiR-N

Code for the paper ["FaiR-N: Fair and Robust Neural Networks for Structured Data"](https://arxiv.org/abs/2010.06113). 
Presented at the 2021 AAAI/ACM Conference on AI, Ethics, and Society (AIES ’21).

We present a novel method for fair and robust training of neural networks on tabular data. We implement a novel distance metric and minimize the average distance in the latent space for the negative prediction label and an underprivilaged protected group. 

Our method: (1) reduces the disparity in the average ability of recourse between individuals in protected groups, and (2) increases the average distance of data points to the boundary to promote adversarial robustness.

The code infrastructure was adopted from Elsayed et al. 2018 (Large margin deep networks for classification) for compatibility with previous work.  

## Running the Code 
To run multiple experiments:
- Modify run_multiple_experiments.py for learning rate, dataset, fairness_wt, robustness_wt, iterations
- the above script loops through the provided parameters by executing run_train.py multiple times

To run a single experiment:
- Modify run_train.py and run 

Results are stored in `save` (sample pkl results are provided for each dataset)
Tensorflow checkpoints are stored in `ckpts`; Note it is setup to delete existing checkpoints, prior to a run

## Datasets
Datasets are found in ./datasets/
Each dataset has a model.py, config.py, and data_provider.py file. 

## Multi-attribute fairness
To utilize multi-attribute fairness, alter the config.py for the datasets.  
For example, the adult dataset, set the feature dictionary in datasets/adult/adult_config.py to:
   self.feature_dict = {'gender': [47, 73], 'race' : [66, 60]} # Multiattribute 

## Working Dependencies:
python 3.7.4
tensorflow-gpu 1.14.0
matplotlib 3.2.1
numpy 1.18.4
pandas 1.0.3
astropy 4.0.1
absl-py 0.9.0

## Citation
```
Shubham Sharma, Alan H. Gee, David Paydarfar, and Joydeep Ghosh. 2021.
FaiR-N: Fair and Robust Neural Networks for Structured Data. In Proceedings
of the 2021 AAAI/ACM Conference on AI, Ethics, and Society (AIES ’21),
May 19–21, 2021, Virtual Event, USA. ACM, New York, NY, USA, 10 pages.
https://doi.org/10.1145/3461702.3462559
```
