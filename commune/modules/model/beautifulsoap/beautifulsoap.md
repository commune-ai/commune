## Model.BeautifulSoap Module

This module is a wrapper over the Beautiful Soap API

#### Sibling Scrap
```bash
c model.beautifulsoap sibling_scrap {url} {keyword}
```

Ex: 
```bash
c model.beautifulsoap sibling_scrap "https://www.ayush.nz/technology" "Introduction"
```

#### Object Scrap
```bash
c model.beautifulsoap object_scrap url={url} params={params}
```

Ex: 
```bash
c model.beautifulsoap object_scrap url="https://www.ayush.nz/technology" params="{title: h5.card-title, content: small.card-text}"
```