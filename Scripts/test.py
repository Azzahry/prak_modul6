import time
import numpy as np
from flask import Flask, request, render_template
from pytorch_tabnet.tab_model import TabNetClassifier

model_path = './tab_predict.zip'
class_list = {
    "Income below 50K" : 0,
    "Income above 50K" : 1
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    loaded = TabNetClassifier()
    loaded.load_model(model_path)

    age = float(request.form['age'])
    edu = float(request.form['education'])
    hpw = float(request.form['hours_per_week'])
    gender = float(request.form['gender'])
    
    start = time.time()

    prob = loaded.predict(np.array([[age, edu, hpw, gender]]))
    runtimes = round(time.time() - start, 3)

    result = prob.tolist()[0]
    pred_label = list(class_list.keys())[result]

    return render_template('/result.html', prediction=pred_label, runtime=runtimes)

if __name__ == '__main__':
    app.run(debug=True)
