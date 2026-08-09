import urllib.request
import os
import json
from PIL import Image

img_path = 'test_sample.jpg'
img = Image.new('RGB', (224, 224), color=(255, 128, 64))
img.save(img_path)

boundary = '---------------------------1234567890'
with open(img_path, 'rb') as f:
    img_bytes = f.read()

body = bytearray()
body.extend(f'--{boundary}\r\n'.encode('utf-8'))
body.extend('Content-Disposition: form-data; name="file"; filename="test_sample.jpg"\r\n'.encode('utf-8'))
body.extend('Content-Type: image/jpeg\r\n\r\n'.encode('utf-8'))
body.extend(img_bytes)
body.extend('\r\n'.encode('utf-8'))
body.extend(f'--{boundary}--\r\n'.encode('utf-8'))

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/predict',
    data=bytes(body),
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

try:
    with urllib.request.urlopen(req) as response:
        print('PREDICTION STATUS:', response.status)
        print('PREDICTION RESULT:', json.dumps(json.loads(response.read().decode('utf-8')), indent=2))
finally:
    if os.path.exists(img_path):
        os.remove(img_path)
