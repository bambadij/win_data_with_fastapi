from fastapi import FastAPI,HTTPException,Response
import joblib
from  pydantic import BaseModel
import pandas as pd
import uvicorn
#load pipeline
pipeline =joblib.load('toolkit/pipeline.joblib')
encoder =joblib.load('toolkit/encoder.joblib')

#print(pipeline)

#Create class to define our input type
class WineFeatures(BaseModel):
    alcohol:float
    malic_acid:float
    ash:float
    alcalinity_of_ash:float
    magnesium:float
    total_phenols:float
    flavanoids:float
    nonflavanoid_phenols:float
    proanthocyanins:float
    color_intensity:float
    hue:float
    od280_od315_of_diluted_wines:float
    proline:float

#CREATE A INSTANCE
app =FastAPI()

@app.get('/')
def home():
    return 'Hello word'

@app.get('/info')
def appinfo():
    return 'This is the info page of this app'

@app.post('/predict_grade')
def predict_wine_grade(wine_features:WineFeatures):
    
    try:
        df = pd.DataFrame([wine_features.model_dump()])
        prediction = pipeline.predict(df)
        
        decoder_prediction = encoder.inverse_transform([prediction])[0]
        
        return {'prediction':decoder_prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500,detail=f'this is a server error contact administrator {str(e)}') 

# if __name__ == '__main__': 
#     uvicorn.run(app)