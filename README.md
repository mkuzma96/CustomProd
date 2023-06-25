# Addressing distributional shifts in Operations Management

Abstract: To meet order fulfillment targets, manufacturers seek to optimize their production plans. Machine learning can support this objective by predicting throughput times on production lines given order specifications. However, this is challenging when manufacturers produce customized products, as customization often leads to changes in the probability distribution of operational data and, thus, so-called distributional shifts. Distributional shifts can harm the performance of predictive models when deployed to future customer orders with new specifications. The Operations Management literature provides little advice on how such distributional shifts can be addressed. Here, we propose a data-driven approach based on adversarial learning and job shop scheduling, which allows us to account for distributional shifts in manufacturing settings with high degrees of product customization. We empirically validate our proposed approach using real-world data from a job shop production that supplies large metal components to an oil platform construction yard. Across an extensive series of numerical experiments, we find that our adversarial learning approach outperforms common baselines and thus offers considerable cost savings. Overall, this paper shows how production managers can improve their decision-making under distributional shifts. 

Paper available at: https://doi.org/10.1111/poms.14021

# Scripts

The repository includes the following (main) scripts:

cross_validation.py (cross validation for implemented method and benchmarks)

prediction.py (prediction results for implemented method and benchmarks)

scheduling.py (scheduling optimization based on prediction results)

tables_plots.ipynb (presentation of results in tables and plots)

# Requirements

python 3.8 

keras 2.9

pulp 2.7

# Experimental details

The three main python scripts are used to reporoduce the results of main experiments in Chapter 5. The variations with diffrent capacities, cost parameters, error distributions, and nonlinearities can be obtained by specifying experimental parameters in the code - the defaults are set for our main experimental result. The folder "data" contains the necessary data for simulation, i.e., estimated distributional moments and functional realtionship between covariates and throughput times from real-world Aker data. Folders "results_pred" and "results_final" contain the output when running the respective python scripts to obtain results. These folders store all of the results, and are used to load the results for presentation in tables and plots. Jupyter notebooks are used for presentation of the results, i.e., plots and tables. The folder "Robustness checks" contains the code for the results in Chapter 6 and the Supplement. The style is the same as for our main scripts. 

# References 

The code implementation of our method was inspired by the following paper:

Shen, J., Qu, Y., Zhang, W., & Yu, Y. (2018). Wasserstein distance guided representation learning for domain adaptation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 32, No. 1).

and Github repositories:

https://github.com/namletien/mcda/blob/83439c526167ce00c0c24d33e742a6d44ae56151/2_visda17/wdgrl.py

https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py


