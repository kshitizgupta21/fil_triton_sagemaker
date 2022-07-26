# XGBoost model inference pipeline with NVIDIA Triton Inference Server on Amazon SageMaker

In this example we show an end-to-end GPU-accelerated fraud detection example making use of tree-based models like XGBoost. In the first notebook [1_prep_rapids_train_xgb.ipynb](1_prep_rapids_train_xgb.ipynb) we demonstrate GPU-accelerated tabular data preprocessing using RAPIDS and training of XGBoost model for fraud detection on the GPU in SageMaker. Then in second notebook [2_triton_xgb_fil_ensemble.ipynb](2_triton_xgb_fil_ensemble.ipynb) we walkthrough the process of deploying data preprocessing + XGBoost model inference pipeline for high throughput, low-latency inference on Triton in SageMaker. 

## Steps to run the notebooks
1. Launch SageMaker **notebook instance** with `g4dn.xlarge` instance.
    - In Additional Configuration select `Create a new lifecycle configuration`. Specify `rapids-2106` as the name in Configuration Setting specify [on_start.sh](on_start.sh) as the lifecycle configuration start notebook script. This will create the RAPIDS kernel for us to use inside SageMaker notebook. 
    - **IMPORTANT:** In Additional Configuration for **Volume Size in GB** specify at least **100 GB**.
    - For git repositories select the option `Clone a public git repository to this notebook instance only` and specify the Git repository URL https://github.com/kshitizgupta21/fil_triton_sagemaker

2. Once JupyterLab is ready, launch the [1_prep_rapids_train_xgb.ipynb](1_prep_rapids_train_xgb.ipynb) notebook with `rapids-2106` conda kernel and run through this notebook to do GPU-accelerated data preprocessing and XGBoost training on credit card transactions dataset for fraud detection use-case.

3. Launch the [2_triton_xgb_fil_ensemble.ipynb](2_triton_xgb_fil_ensemble.ipynb) notebook using `conda_python3` kernel and run through this notebook to deploy the ensemble data preprocessing + XGBoost model inference pipeline using the Triton's Python and FIL Backends on Triton SageMaker `g4dn.xlarge` endpoint.
