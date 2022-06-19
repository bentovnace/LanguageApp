import base64
import json                    

import requests

api = 'http://192.168.1.14:6868/etext'
image_file = 'Capture_2.JPG'

with open(image_file, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
  
payload = json.dumps({"image": im_b64, "other_key": "value"})
response = requests.post(api, data=payload, headers=headers)

print(response.text)