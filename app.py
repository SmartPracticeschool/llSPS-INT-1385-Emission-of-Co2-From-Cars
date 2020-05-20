import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index1.html', prediction_text='Emission of Co2 is :  {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    threshold = 250
    if prediction > threshold:
        return render_template('index1.html', prediction_text= 'Threshold limit exceeded by : {}'.format(prediction-threshold))
    else:
        return render_template('index1.html', prediction_text='Emission of Co2 is  {} and is under control'.format(output))
    output = prediction[0]+""
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
