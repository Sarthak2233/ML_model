# Import modules and packages
from flask import (
    Flask,
    request,
    render_template,
    url_for
)
import pickle
import numpy as np
from scipy.spatial import distance

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def get_input_values():
    val = request.form['my_form']


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return 'The URL /predict is accessed directly. Go to the main page firstly'

    if request.method == 'POST':
        input_val = request.form

        if input_val != None:
            # collecting values
            vals = []
            for key, value in input_val.items():
                vals.append(float(value))

        # Calculate Euclidean distances to freezed centroids
        with open('freezed_data.pkl', 'rb') as file:
            freezed_centroids = pickle.load(file)

        for key, value in freezed_centroids.items():
            if key == 'centroids':
                centroids = value
            if key == 'inertia':
                inertia = value
            if key == 'silhoutte_score':
                silhoutte_score = value

        assigned_clusters = []
        l = []  # list of distances

        for i, this_segment in enumerate(freezed_centroids['centroids']):
            print(this_segment)
            dist = distance.euclidean(vals, this_segment)
            l.append(dist)
            index_min = np.argmin(l)
            assigned_clusters.append(index_min)


        return render_template(
            'predict.html',
            result_value=f'According form the model the inputs will behaves as form the datas that are inluded form the  #{index_min} cluster. \n ',
            inertia_value=f'The inertia to this given model is: {inertia}.',
            silhoutte_score=f"The silhoutte score to this created model is {silhoutte_score}"
        )

@app.route('/predict1', methods=['POST', 'GET'])
def predict1():
    if request.method == 'POST':
        input_val = request.form
        if input_val != None:
            val = []
            for i, j in input_val.items():
                val.append(float(j))
            features = [np.array(val)]
            print(features)
            with open('linearmodel.pkl', 'rb') as f:
                contents = pickle.load(f)
            for key, value in contents.items():
                if key == 'linearmodel':
                    result = value.predict(features)
                if key == 'MeanSquaredError':
                    mse = value
                else:
                    r2 = value

    return render_template(
        'predict.html',
        result = f'{round(result[0],2)} amount of quantity will be likely to sell.',
        mean_squared_error = f'Mean Squared error for this linear model to predict quantity is {mse}',
        accuracy = f'Accuracy for this model is {r2}'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)