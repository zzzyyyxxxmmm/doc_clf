from flask import Flask, request, render_template, jsonify, make_response,flash,redirect,url_for
from flask_restful import Api, Resource, reqparse
import json
import numpy as np
from sklearn.externals import joblib
import pickle

application = Flask(__name__)

application.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
application.static_folder = 'static'
api = Api(application)

def init():
    global encoder
    global clf
    encoder_file = open("models/encoder.pk", "rb")
    encoder = pickle.load(encoder_file)
    clf = joblib.load("models/clf_RandomForestClassifier.joblib")
    encoder_file.close()

@application.route("/")
def home():
    init()
    return render_template("index.html")


@application.route("/predict")
def predict():
    return render_template("predict.html", title="Predict")


@application.route("/about")
def about():
    return render_template("about.html", title="About")


class PredictAPI(Resource):

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('words', type=str, location='json')
        super(PredictAPI, self).__init__()

    def get(self):
        init()
        # print(request.args.get("words"))
        words2feature = encoder.transform([request.args.get("words")])
        return {'label': clf.predict(words2feature)[0]}

    def post(self):
        init()
        input = request.form["words"]
        words2feature = encoder.transform([input])
        result=str(clf.predict(words2feature)[0])
        flash(f'Result: {result}!', 'success')
        return redirect(url_for('predict'))


class MuiltiplePredictAPI(Resource):

    def __init__(self):
        super(MuiltiplePredictAPI, self).__init__()

    def post(self):
        init()
        request_json_data = request.get_json(force=True)
        words2feature = encoder.transform(request_json_data['words'])
        accuracy=clf.score(words2feature,request_json_data['labels'])
        return {"accuracy":accuracy,'label': str(clf.predict(words2feature))}

    def get(self):
        pass


api.add_resource(PredictAPI, '/api/v1.0/predict', endpoint='getwords')
api.add_resource(PredictAPI, '/predict', endpoint='postwords')
api.add_resource(MuiltiplePredictAPI, "/api/v1.0/predictmore", endpoint='getmore')


@application.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == "__main__":
    application.run()
