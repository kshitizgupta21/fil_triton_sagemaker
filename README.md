Steps

1. Launch SageMaker notebook instance with g4dn.xlarge instance.
    - In additional configuration specify on_start.sh as the lifecycle configuration start notebook script. This will create the RAPIDS kernel for us to use inside SageMaker notebook. 
    - 200GB storage should be fine.
    - For git repository specify https://github.com/kshitizgupta21/fil_triton_sagemaker

2. Once JupyterLab is ready, launch the 1_prep_rapids_train_xgb.ipynb notebook with rapids2204 (rapids2204 has a temporary issue so for now use rapids2206) kernel. Run through this notebook.

3. Run through 2_deploy_triton.ipynb to deploy the ensemble model on Triton SageMaker endpoint.
