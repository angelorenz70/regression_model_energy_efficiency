from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()
    xdata = np.array(list(data.values())).astype(float).reshape(1, 8)

    file_cooling = open('enerygy_efficiency/establishment_energy_efficiency_cooling.pickle', 'rb')
    model1 = pickle.load(file_cooling)
    file_heating = open('enerygy_efficiency/establishment_energy_efficiency_heating.pickle', 'rb')
    model2 = pickle.load(file_heating)

    # make a prediction using the input values and the loaded model
    y_pred_cooling = model1.predict(xdata)
    y_pred_heating = model2.predict(xdata)

    # Make the prediction
    prediction_cooling = y_pred_cooling[0]
    prediction_heating = y_pred_heating[0]

    # Return the prediction as a JSON response
    return jsonify({'prediction1': prediction_cooling, 'prediction2': prediction_heating})

if __name__ == '__main__':
    app.run(debug=True)



