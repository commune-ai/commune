
import requests

def graphql_query(query, url):
    # endpoint where you are making the request
    print(url, 'da fuck')
    request = requests.post(url,
                            '',
                            json={'query': query})
    if request.status_code == 200:
        request_json = request.json()
        if 'data' in request_json:
            return request_json['data']
        else:
            raise Exception('There was an error in the code fam {}', request_json)
    else:
        raise Exception('Query failed. return code is {}.      {}'.format(request.status_code, query))
