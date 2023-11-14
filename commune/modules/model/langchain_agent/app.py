from flask import Flask, request, jsonify
from agent.agent import WebSurfingAgent

app = Flask(__name__)

agent = WebSurfingAgent()

@app.route('/search', methods=['POST'])
def search():
    # Extract the query from the POST request
    data = request.get_json()
    query = data.get('query', '')

    # Use the agent to perform the search
    results = agent.search_the_web(query)

    return jsonify(results)
    # return results


if __name__ == '__main__':
    app.run(debug=True)
