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

#### Password bypassing
```bash
c model.beautifulsoap password_scrap url={url} cred={credentails} params={params}
```

Ex: 
```bash
c model.beautifulsoap password_scrap url=http://testphp.vulnweb.com/userinfo.php cred="{uname: test, pass: test}" params="{username: input[name='urname']}"
```

#### Image scraper
Scraped images will be saved into images/images directory of root.
```bash
c model.beautifulsoap image_scrap url={url}
```

Ex: 
```bash
c model.beautifulsoap image_scrap https://www.ayush.nz/technology
```

#### Generate random url
```bash
c model.beautifulsoap generate_random_url
```

#### Get all buttons and inputs from the website
```bash
c model.beautifulsoap get_buttons_and_inputs url={url}
```

Ex:
```bash
c model.beautifulsoap get_buttons_and_inputs https://blockchain.news/
```

#### Get all components based on user input
```bash
c model.beautifulsoap get_components url={url} tags={tags}
```

Ex:
```bash
c model.beautifulsoap get_components https://blockchain.news/ tags="["button", "input", "h1"]"
```