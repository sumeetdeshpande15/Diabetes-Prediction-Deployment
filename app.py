# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        blood_pressure = int(request.form['bloodpressure'])
        bloodpressure_category = int(request.form['bloodpressure_Category'])
        skin_thickness = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        insulin_category = int(request.form['Insulin_Category'])
        bmi = float(request.form['bmi'])
        bmi_category = int(request.form['BMI_Category'])
        diabetes_pedigree_function = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[pregnancies, glucose, blood_pressure, bloodpressure_category, 
                          skin_thickness, insulin, insulin_category, bmi, 
                          bmi_category, diabetes_pedigree_function, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)