## Nonnegative Matrix Factorization
NMF is a python program that applies different nonnegative matrix factorization algorithms for clustering.  
Currently, this program supports
  * Multiplicative Updates (MU)[1](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf)  
  * Alternating Least Squares (ALS)[2](https://www.sciencedirect.com/science/article/pii/S0167947306004191)  
  * Alternating Nonnegative Least Squares with Active Set (ANLS - AS)[3](https://www.cc.gatech.edu/~hpark/papers/simax-nmf.pdf)  

### Results
Experimental results with abcnews-date-test.csv's headline_text  
Multiplicative Updates (MU):  
<img src="/experimental_results/abc_mu.png">  
Alternating Least Squares (ALS):  
<img src="/experimental_results/abc_als.png">  

Experimental results with bbcsports.csv and k = 5:  

[Data](https://github.com/hpark95/Nonnegative-Matrix-Factorization/tree/master/experimental_results/k5/data)  
[Clusters generated](https://github.com/hpark95/Nonnegative-Matrix-Factorization/tree/master/experimental_results/k5/clusters)


Expertimental results for k = 100:  

[Data](https://github.com/hpark95/Nonnegative-Matrix-Factorization/tree/master/experimental_results/k100/data)  
[Clusters generated](https://github.com/hpark95/Nonnegative-Matrix-Factorization/tree/master/experimental_results/k100/clusters)

### How to Run
* Usage: main.py [-h] -f FILENAME -c COL_NAME -m {sklearn,all,als,anls,mu}
               [-d DATA_FRAC] [-r RANDOM_SAMPLE] [-n NUM_MAX_FEATURE]
               [-s CLUSTER_SIZE] [-k NUM_CLUSTERS] [-i NUM_ITERS]
               [-p PRINT_ENABLED]  

* Required arguments:  
  * -f FILENAME, --filename FILENAME  
    the input file name  
  * -c COL_NAME, --col_name COL_NAME  
    the column of the input csv file for nonnegative matrix factorization.  
  * -m {sklearn,all,als,anls,mu}, --method {sklearn,all,als,anls,mu}  
    the NMF method to apply  

* Optional arguments:  
  * -h, --help  
    show this help message and exit  
  * -d DATA_FRAC, --data_frac DATA_FRAC  
    the amount of the data to be used  
  * -r RANDOM_SAMPLE, --random_sample RANDOM_SAMPLE  
    if set False, disables random sampling of the data  
  * -n NUM_MAX_FEATURE, --num_max_feature NUM_MAX_FEATURE  
    the maximum number of features to be discovered in the dataset  
  * -s CLUSTER_SIZE, --cluster_size CLUSTER_SIZE  
    the number of features in each cluster  
  * -k NUM_CLUSTERS, --num_clusters NUM_CLUSTERS  
    the number of clusters to be discovered  
  * -i NUM_ITERS, --num_iters NUM_ITERS  
    the number of iterations to run a NMF algorithm  
  * -p PRINT_ENABLED, --print_enabled PRINT_ENABLED  
    if ture, output print statements  

### Citation
Algorithms for Non-negative Matrix Factorizations by D. Lee and H. Seung,  
https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf  
Algorithms and applications for approximate nonnegative matrix factorization by M. Berry,  
https://www.sciencedirect.com/science/article/pii/S0167947306004191  
Non-negative Matrix Factorization Based on Alternating Non-negativity Constrained Least Squares and ActiveSet Method by H. Kim and H. Park,  
https://www.cc.gatech.edu/~hpark/papers/simax-nmf.pdf  
