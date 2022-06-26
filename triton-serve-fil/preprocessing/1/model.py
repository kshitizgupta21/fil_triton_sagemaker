import pandas as pd
import os
import sklearn
import triton_python_backend_utils as pb_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import pickle
from pathlib import Path

COL_NAMES = ['User',
 'Card',
 'Year',
 'Month',
 'Day',
 'Time',
 'Amount',
 'Use Chip',
 'Merchant Name',
 'Merchant City',
 'Merchant State',
 'Zip',
 'MCC',
 'Errors?']

CAT_COL_NAMES = ['Zip', 'MCC', 'Merchant Name', 'Use Chip', 'Merchant City', 'Merchant State']

us_states_plus_online = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY', 'ONLINE']

LABEL_ENCODERS_FILE = 'label_encoders.pkl'

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model config
        
        self.model_config = json.loads(args['model_config'])
        
        output_config = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT")


        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])
        
        cur_folder = Path(__file__).parent
        with open(str(cur_folder/LABEL_ENCODERS_FILE), 'rb') as f:
                  self.encoders = pickle.load(f)
        
        
    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        
        for request in requests:
            # Get input tensors 
            input_data = pb_utils.get_input_tensor_by_name(request, 'INPUT').as_numpy()
            print("input data is", input_data)
            print("type of input_data is", type(input_data))
            
            # we get byte representation so we convert it to to string
#             if is_string_dtype(data.dtype):
#                 data = data.astype("str")
#             # in case of string data cudf can only create dataframe from list of data
#             data = cudf.DataFrame(input_data.tolist(), columns=COLUMN_NAMES)
            
#             data['Amount'] = data['Amount'].str.slice(1)
#             data.loc[data["Merchant City"]=="ONLINE", "Merchant State"] = "ONLINE" 
#             data.loc[data["Merchant City"]=="ONLINE", "Zip"] = "ONLINE" 
#             data['Errors?'] = data['Errors?'].notna().astype(int)
        
#             data.loc[~data["Merchant State"].isin(us_states_plus_online), "Zip"] = "FOREIGN"
#             data['Amount'] = data['Amount'].str.slice(1).astype('float32')
#             data['Hour'] = data['Time'].str.slice(stop=2).astype('int64')
#             data['Minute'] = data['Time'].str.slice(start=3).astype('int64')
#             data.drop(columns=['Time'], inplace=True)

#             for col in CAT_COL_NAMES:
#                 le = LabelEncoder()
#                 le.classes_ = self.encoders[col]
#                 data[col] = le.transform(data[col])
                
#             # Create output tensors. You need pb_utils.Tensor
#             # objects to create pb_utils.InferenceResponse.
            
#             data = data.astype(self.output_dtype)
#             data_np = data.values_host
            
#             data_tensor = pb_utils.Tensor(
#                 'OUTPUT',
#                 data_np.astype(self.output_dtype))


#             # Create InferenceResponse. You can set an error here in case
#             # there was a problem with handling this inference request.
#             # Below is an example of how you can set errors in inference
#             # response:
#             #
#             # pb_utils.InferenceResponse(
#             #    output_tensors=..., TritonError("An error occured"))
#             inference_response = pb_utils.InferenceResponse(output_tensors=[data_tensor])
#             responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses


    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
