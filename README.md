<h1 align="center">ITE-BE</h1>


## About

Quantum algorithms can be used to solve classically hard optimization problem such as the MaxCut problem. In [our paper](https://arxiv.org/abs/2411.10737), we adopt a [recently developed implementation (ITE-BE)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.7.013306) of imaginary time evolution using exact block encoding to solve for the MaxCut problem. We also manage to integrate the ITE-BE approach with quantum approximate optimization algorithm (QAOA). The source code of these quantum solver for the MaxCut problem is in the `imaginary_time_optimization`.  

Since the original comprehensive data is large, in the `data` folder we only include the python code to solve for MaxCut problem of unweight 3-regular graphs (u3R) and bipartite graphs rather than uploading real data files. 

## Contact
If you have any questions or suggestions, please contact via email [daweiz@usc.edu](mailto:daweiz@usc.edu) or create an Github issue in this repository. 
