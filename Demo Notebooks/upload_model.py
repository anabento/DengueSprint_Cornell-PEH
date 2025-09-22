import os
import numpy as np
import pandas as pd
from epiweeks import Week
import mosqlient as mosq
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# First upload the model and retrieve the model_id!
# Stop the code after the 'upload model' block
# Then insert the correct ID in the 'upload forecast' block and COMMENT OUT ALL CODE IN THE UPLOAD MODEL BLOCK

##################
## Load API key ##
##################

# Load environment variables from a specific file
dotenv_path = os.path.join(os.getcwd(), '..', 'API_KEY.env')
load_dotenv(dotenv_path=dotenv_path)

# Get the API key
api_key = os.getenv('API_KEY')


##################
## Upload model ##
##################

# name = "Cornell PEH - NegBinom Baseline model"
# description = "Negative Binomial baseline model for the 2025 sprint"
# repository = "https://github.com/anabento/DengueSprint_Cornell-PEH"
# implementation_language = "Python"
# disease = "dengue"
# temporal = True
# spatial = True
# categorical = False
# adm_level = 1
# time_resolution = "week"
# sprint = True

# model = mosq.upload_model(
#     api_key=api_key,
#     name=name,
#     description=description,
#     repository=repository,
#     implementation_language=implementation_language,
#     disease=disease,
#     temporal=temporal,
#     spatial=spatial,
#     categorical=categorical,
#     adm_level=adm_level,
#     time_resolution=time_resolution,
#     sprint=sprint
# )

# print(model.dict())

# import sys
# sys.exit()

#####################
## Upload forecast ##
#####################

# set the correct model_id
model_id = 139

# define validation experiment indices..
validation_indices = [1,2,3]

# .. and loop over them
for validx in validation_indices:
    # load validation experiment data
    forecast = pd.read_csv(f'../data/interim/baseline_model-validation_{validx}.csv', index_col=0)
    # get the ufs..
    ufs = forecast['adm_1'].unique().tolist()
    # ..and loop over them
    for uf in ufs:
        # slice data
        df = forecast[forecast['adm_1'] == uf].reset_index()
        # push the prediction
        res = mosq.upload_prediction(
            model_id = model_id, 
            description = f'Validation {validx} (NegBinom Baseline model)', 
            commit = '01fa194bf4ca05759f2c370d0cb2971e2fb4adc0',
            predict_date = '2025-07-29', 
            prediction = df,
            adm_1=f'{uf}',
            api_key = api_key) 