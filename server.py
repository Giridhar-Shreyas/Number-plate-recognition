from flask import Flask, request, jsonify
from PIL import Image
import train as train

app = Flask(__name__)
app.app_context().push()

@app.route('/', methods=['POST'])
def get_img():
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file.stream)
        bbox = train.predict(img)
        return jsonify({'x': bbox[0][0].item(),'y': bbox[0][1].item(),'w': bbox[0][2].item(), 'h': bbox[0][3].item()})





app.run(host="0.0.0.0", port=80)