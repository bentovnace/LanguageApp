from flask import Flask, request, jsonify,abort
import requests
import json
import PIL.Image as Image
import PIL.Image 
from PIL import Image
import io
import base64
import numpy as np 
import pytesseract
from chat import start_chat
from detect import run 
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR\\tesseract.exe'
from flask_cors import CORS, cross_origin

app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def recognition_text(img):
    text = pytesseract.image_to_string(img)
    print(text)
    
    if text == '' :
        print('error')
    else:
        print('OK')
    return text

#def base64_to_img(base64data):
#    base64_data =base64.b64decode(base64data)
#    filename = "image.jpg"
#    with open(filename,'wb') as f:
#        f.write(base64_data)
#    return "image.jpg"




@app.route('/ebody', methods=['GET','POST'])
@cross_origin(origin='*')
def Bodylanguage_recognition():
    #base64_data_pro = request.form.get('base64_code')
    img_read = base64_to_img() #(base64_data_pro)
    return str(run(img_read))



@app.route('/etext', methods=['GET','POST'])
@cross_origin(origin='*')
def text_recogniton():
    #base64_data_pro = request.form.get('base64_code')  
    img_read = base64_to_img() #(base64_data_pro)  
    return str(recognition_text(img_read))




@app.route('/ebot', methods=['GET','POST'])
@cross_origin(origin='*')
def Chat_bot():
    Input = request.args.get('input') 
    chatInput = str(Input)
    return str(start_chat(chatInput))

@app.route("/test", methods=['POST'])
def base64_to_img():         
       
    if not request.json or 'image' not in request.json: 
        abort(400)         
    im_b64 = request.json['image']
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    filename = "image.jpg"
    with open(filename,'wb') as f:
        f.write(img_bytes)
    img = Image.open(io.BytesIO(img_bytes))

    img_arr = np.asarray(img)      
    print('img shape', img_arr.shape)

   

    
    return "image.jpg"

@app.route('/bodyimg', methods=['GET','POST'])
def body_img():
    image_file = 'bodylang.jpg'
    with open(image_file, "rb") as f:
        im_bytes = f.read()        
    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    return str(im_b64)
if __name__  == "__main__":
    app.run(host ='0.0.0.0',port = '6868')
