Steps

1. Launch SageMaker notebook instance with g4dn.xlarge instance.
    - In additional configuration specify on_start.sh as the lifecycle configuration start notebook script. This will create the RAPIDS kernel for us to use inside SageMaker notebook. 
    - 200GB storage should be fine.
    - For git repository specify https://github.com/kshitizgupta21/fil_triton_sagemaker

2. Once JupyterLab is ready, launch the 1_prep_rapids_train_xgb.ipynb notebook with rapids2106 conda kernel. Run through this notebook.

3. Launch the 2_triton_xgb_fil_ensemble.ipynb notebook using conda_python3 kernel (we don't needs RAPIDS in this notebook) and run through this notebook to deploy the ensemble FIL model on Triton SageMaker endpoint.
