import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = "model_RandomForest.bin"


with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("noshow")

@app.route("/predict", methods=["POST"])
def predict():

    patient = request.get_json()

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]

    result = {
        "no_show_probability": float(y_pred)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)