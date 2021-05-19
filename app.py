import joblib
import numpy as np
from flask import Flask,request,render_template

app = Flask(__name__)
IRIS_Model = joblib.load('IRIS_MODEL.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    output = ' '
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print (final_features)
    prediction = IRIS_Model.predict(final_features)
    output = round(prediction[0],2)
    # print(prediction[0])

    # if output == 0:
    #     outpu = 'Iris-setosa'
    # elif  output == 1:  
    #      output = 'Iris-versicolor'
    # else:
    #      output = 'Iris-virginica'   

    return render_template('index.html',prediction_text='The IRIS Flower is:{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)