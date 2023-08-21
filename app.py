from click import style
from flask import Flask, send_file, jsonify, render_template, request, send_file
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import math


fig,ax = plt.subplots(figsize=(6,6))
ax=sns.set_style(style="darkgrid")

x=[i for i in range(100)]
y=[i for i in range(100)]


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def predict():

    store=float(request.form['store'])
    item=float(request.form['item'])
    year= float(request.form['year'])
    month=float(request.form['month'])
    day_of_year=float(request.form['day_of_year'])
    
    
    
    X= np.array([[ store, item, year, month, day_of_year ]])
    
    model_path=r'C:\Users\Samsung\Desktop\Sales-Prediction-Flask-Deployement\models\cat_model.sav'

    model= joblib.load(model_path)

    Y_pred=model.predict(X)
    

    return render_template("predict.html", my_prediction = math.floor(Y_pred))


@app.route('/visualize',methods=['POST','GET'])
def visualize():

    store=1
    item=2

    forecastCAT=pd.read_csv(r'C:\Users\Samsung\Desktop\Sales-Prediction-Flask-Deployement\datasets\file.csv')

    plot1=[[(forecastCAT.store==int(store)) & (forecastCAT.item==int(item))]]['sales']
    plot1.plot(color = "orange", figsize = (25,10),legend=True,label="Store "+str(store)+" item "+str(item)+" forecast")
    canvas=FigureCanvas(fig)
    img=io.BytesIO()
    fig.savefig(img)
    img.seek(0)

    return render_template("visualize.html", visualize = send_file(img,mimetype='img/png'))


if __name__ == "__main__":
    app.run(debug=True)

