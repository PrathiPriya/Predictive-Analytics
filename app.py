import pandas as pd
from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)

@app.route("/")
def loadPage():
    return render_template('Home.html', query1="", query2="", query3="", query4="", query5="", query6="",
                           query7="", query8="", query9="", query10="", query11="", query12="", query13="",
                           query14="", query15="", query16="", query17="", query18="", query19="")

@app.route("/", methods=['POST'])
def predict_churn():
    input_features = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'Gender', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'PaymentMethod', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'Tenure'
    ]

    # Collect input values from the form
    input_values = [request.form[feature] for feature in input_features]

    # Convert input values to numeric
    input_values = [float(value) if value.isdigit() else 0.0 for value in input_values]

    # Load the churn prediction model using joblib
    model = load('Churn_model.joblib')

    # Make predictions
    prediction = model.predict([input_values])[0]
    probability = model.predict_proba([input_values])[0][1]

    # Customize the output messages based on the prediction
    if prediction == 1:
        output1 = "This customer is likely to churn."
        output2 = "Confidence: {:.2f}%".format(probability * 100)
    else:
        output1 = "This customer is likely to continue."
        output2 = "Confidence: {:.2f}%".format((1 - probability) * 100)

    return render_template('churn_home.html', output1=output1, output2=output2,
                           query1=float(request.form['SeniorCitizen']),
                           query2=float(request.form['MonthlyCharges']),
                           query3=float(request.form['TotalCharges']),
                           query4=request.form['Gender'],
                           query5=request.form['Partner'],
                           query6=request.form['Dependents'],
                           query7=request.form['PhoneService'],
                           query8=request.form['MultipleLines'],
                           query9=request.form['InternetService'],
                           query10=request.form['OnlineSecurity'],
                           query11=request.form['OnlineBackup'],
                           query12=request.form['DeviceProtection'],
                           query13=request.form['TechSupport'],
                           query14=request.form['PaymentMethod'],
                           query15=request.form['StreamingTV'],
                           query16=request.form['StreamingMovies'],
                           query17=request.form['Contract'],
                           query18=request.form['PaperlessBilling'],
                           query19=float(request.form['Tenure']))

if __name__ == "__main__":
    app.run(debug=True)
