import os
import pandas as pd
import mosqlient as mosq
from datetime import date
from dotenv import load_dotenv

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


##################################
## (Run only ONCE) Upload model ##
##################################

# name = "Cornell PEH - NegBinom Baseline"
# description = "Negative Binomial baseline model for the 2025 sprint. This model squashes a bug found in model 139. Authored by Tijs W. Alleman & Ana I. Bento."
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
model_id = 158

# define validation experiment indices..
validation_indices = None # (forecast), or [1, 2, 3] (validation)

# validation
if validation_indices:
    # .. and loop over them
    for validx in validation_indices:
        # set correct ID and description
        ID = f"baseline_model-validation_{validx}"
        description = f"Validation {validx} (Cornell PEH - NegBinom Baseline model). Authored by Tijs W. Alleman & Ana I. Bento."
        commit = "4f65fd7af1bfa9c1a9b86469a798179b666e2be7"
        # load validation experiment data
        forecast = pd.read_csv(f'../data/interim/{ID}.csv', index_col=0)
        # get the ufs..
        ufs = forecast['adm_1'].unique().tolist()
        # ..and loop over them
        for uf in ufs:
            # slice data
            df = forecast[forecast['adm_1'] == uf].reset_index()
            # push the prediction
            res = mosq.upload_prediction(
                model_id = model_id, 
                description = description, 
                commit = commit,
                predict_date = str(date.today()), 
                prediction = df,
                adm_1=f'{uf}',
                api_key = api_key) 
# forecast
else:
    # set correct ID and description
    ID = f'baseline_model-forecast'
    description = f'Forecast (Cornell PEH - NegBinom Baseline model). Authored by Tijs W. Alleman & Ana I. Bento.'
    commit = 'e90b5c07283944dcdd0dedc404aff8fcce8146ce'
    # load validation experiment data
    forecast = pd.read_csv(f'../data/interim/{ID}.csv', index_col=0)
    # get the ufs..
    ufs = forecast['adm_1'].unique().tolist()
    # ..and loop over them
    for uf in ufs:
        # slice data
        df = forecast[forecast['adm_1'] == uf].reset_index()
        # push the prediction
        res = mosq.upload_prediction(
            model_id = model_id, 
            description = description, 
            commit = commit,
            predict_date = str(date.today()), 
            prediction = df,
            adm_1=f'{uf}',
            api_key = api_key) 