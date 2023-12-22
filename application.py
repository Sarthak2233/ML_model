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

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/', methods=['POST'])
def get_input_values():
    val = request.form['my_form']


@application.route('/predict', methods=['POST', 'GET'])
def predict():
    global index_min
    global centroids
    global inertia, silhoute_score
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

        print(freezed_centroids)

        for key , value in freezed_centroids.items():
            if key=='centroids':
                centroids=value
            if key=='inertia':
                inertia = value
            if key=='silhoutte_score':
                silhoutte_score = value

        print(centroids)
        print(inertia)

        assigned_clusters = []
        l = []  # list of distances

        for i, this_segment in enumerate(centroids):
            dist = distance.euclidean(vals, this_segment)
            l.append(dist)
            index_min = np.argmin(l)
            assigned_clusters.append(index_min)

        return render_template(
            'predict.html',
            result_value=f'According form the model the inputs will behaves as form the datas that are inluded form the  #{index_min} cluster. \n ',
            inertia_value =  f'The inertia to this given model is: {inertia}.',
            silhoutte_score = f"The silhoutte score to this created model is {silhoutte_score}"
            )

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=80, debug=True)