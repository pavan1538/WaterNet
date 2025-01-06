from flask import Flask, render_template, request
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Machine learning model (training data for demonstration)
model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train = np.array([[7.0, 2.5, 8.0, 500], [8.5, 5.0, 6.5, 700]])
y_train = np.array([1, 0])
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_quality', methods=['POST'])
def check_quality():
    pH = float(request.form['pH'])
    turbidity = float(request.form['turbidity'])
    do = float(request.form['do'])
    conductivity = float(request.form['conductivity'])

    prediction = model.predict([[pH, turbidity, do, conductivity]])
    result = "Suitable for Drinking and Irrigation" if prediction[0] == 1 else "Not Suitable for Drinking or Irrigation"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
