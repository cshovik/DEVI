from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Replace with Laptop 1's IP address and port
LAPTOP_1_IP = "http://192.168.1.10:5000"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_sos', methods=['GET'])
def check_sos():
    try:
        response = requests.get(LAPTOP_1_IP + "/sos_status")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)