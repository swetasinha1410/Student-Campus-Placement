# Library imports
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
import os
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder,MinMaxScaler , StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC




# Load trained Pipeline
model= joblib.load(open("student_placement_v1.0.model","rb"))

# Create the app object
app = Flask(__name__)

# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_values = [np.array(input_features)]
    
    features_name = ['Gender','Specialisation','Techinal Degree','Work Experience','SSC_p','High School_p','Degree_p','MBA_p']
    
    df=pd.DataFrame(features_values,columns=features_name)
    predictions =model.predict(df)
 
    if predictions==0:
        return render_template('index.html', prediction_text='**Not Placed**')
    else:
        return render_template('index.html', prediction_text='**Placed**')


if __name__ == "__main__":
    app.run(debug=True)
