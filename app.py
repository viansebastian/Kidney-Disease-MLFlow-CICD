import os
from src.cnn_classifier.utils.common import decode_image
from src.cnn_classifier.pipeline.prediction import PredictionPipeline
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app) 

class ClientApp: 
    def __init__(self):
        self.filename = 'inputImage.jpg'
        self.classifier = PredictionPipeline(self.filename)
    
    
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system('dvc repro')
    return 'Training successful'

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decode_image(image, client_app.filename)
    result = client_app.classifier.predict()
    return jsonify(result)

if __name__ == '__main__':

    client_app = ClientApp() 
    app.run(host="0.0.0.0", port=8080)