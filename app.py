import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__, template_folder= 'template') #create flask app
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict',methods = ['POST']) #model will provide output for inputs

def predict():


    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text = 'Heart_disease: (1 = Yes, 0 = No) {}'.format(output))

if __name__== "__main__":
    app.run(debug= True)
