
class Endpoint:
    def _endpoint(self, method, path, **kwargs):
        return self._request(method, path, **kwargs)