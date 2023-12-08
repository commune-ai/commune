from flask import Flask, request
from ethereum.blockchain import process_transaction

app = Flask(__name__)

@app.route('/send_transaction', methods=['POST'])
def send_transaction():
    data = request.json
    return process_transaction(data)

if __name__ == 'main':
    app.run(debug=True)