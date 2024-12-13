# SamplingIon-Insertion

Stochastic insertion of a counter ion into crystal pores.

The idea is to use a Monte Carlo simulation to sample and determine the most likely position to find a counterion, this was specially designed to work with Metal-Organic Frameworks.

## General steps of the algorithm

1. The structure of the MOF and the ion is read.

2. First, the MOF is analyzed. This is started by calling the `DrunkenIon` class. 



n. Print final report

## How to use

Central command, this will generate a trajectory of possible positions:

```
drunkenIon -i sql_v1-4X_Ni_1-2X_N3.cif --run_sampling -n_steps 50000 -T 5.0 --show_plots -step_size 5.0
```

Search and study of pores. It is always considered that there are between 2 to 11 pores and a study is made to know which number fits better. This is saved in the sampling.pkl file for future studies.

```
drunkenIon -load sampling.pkl --cluster_study -n_cpu 1 -n_porous 9
drunkenIon -load sampling.pkl --hist_study -n_porous 4 --show_plot
```

Finally, the desired number of ions is added.
```
drunkenIon -load sampling.pkl -ion ion_I.xyz -n_ions 4
```





