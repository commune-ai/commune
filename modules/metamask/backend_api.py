from flask import Flask, request
from ethereum.blockchain import process_transaction, make_uniswap_trade, get_transaction_history

app = Flask(__name__)

@app.route('/send_transaction', methods=['POST'])
def api_send_transaction():
    data = request.json
    return process_transaction(data)

@app.route('/make_trade', methods=['POST'])
def api_make_trade():
    data = request.json
    return make_uniswap_trade(data)

@app.route('/transaction_history', methods=['GET'])
def api_transaction_history():
    address = request.args.get('address')
    return get_transaction_history(address)

if __name__ == '__main__':
    app.run(debug=True)