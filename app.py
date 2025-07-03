from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

pipeline = pickle.load(open('model/pipeline.pkl', 'rb'))
columns_name = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data], columns=columns_name)
        prediction = pipeline.predict(input_data)[0]
        result = {
            'prediction': int(prediction),
            'message': 'Passenger Survived' if prediction == 1 else 'Passenger Died'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)