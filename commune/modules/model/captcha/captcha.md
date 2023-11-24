
# **Stock**

## Setting a `https://api.capmonster.cloud/` Key

To set a `https://api.capmonster.cloud/` key, use the following command:

```bash
c model.captcha add_api_key YOUR_API_KEY
```

Once this is done, you won't need to specify the key every time you start your module. The key will be saved in the module's folder at `.commune/model.captcha/api_keys.json`.

To remove an API key, use:

```bash
c model.captcha rm_api_key YOUR_API_KEY
```

To list all API keys:

```bash
c model.captcha api_keys
```

## Serving the Model

To serve the model with a specific tag and API key, use the following command:

```bash
c model.captcha serve tag=10 api_key=....
```

## Registering the Model

To register the model with a specific tag and name, use:

```bash
c model.captcha register api_key=.... tag=10
```

This will set the name to `model.captcha::10`.

## Testing the Model

To test the model, use:

```bash
c model.captha recaptcha2_proxyless website_url="https://lessons.zennolab.com/captchas/recaptcha/v2_simple.php?level=high" website_key="6Lcg7CMUAAAAANphynKgn9YAgA4tQ2KI_iqRyTwd"
```

**Response:**
```
{
    'gRecaptchaResponse': 
'03AFcWeA55pXlbxgXjzmKbAMBrvnYUUWPK2JqMFuR2PGNbxh23oiIIuSpPLSqhLnd-uvkBINEdIGJcE0F-j8fbVWu0ryFnNml0DSy-ImzucS2JvoRX7xG8wu7BtzEbk40sAsbrsNTbCBoKJuJM9dQQRcpphYLgLYJDOVVjN6Gszm5FaCunyRvBe0LUN
x8bYfnXnf_5jrC8xDdpK8RCP46g4buAPZKh0kCiECXcXn8UgWrgjZVdxQ2TLeuQsDqW2zgcW1ToKA-QPJkEkEw2jxO1t50-OK7qxvau4EzwWEZBXxYgybp_7SZiWdSJNcetwFajXo3NAJO7LBbarMLgS-N2CV179qaa7pXIVKO9nCiQ2siXiw_MvunsR
7X7aIuH0uN61qoXMusTZZnqVyCHdpZBAgT0Za7ES3bBpFYUgifSdRLl7BVj6Mt4-pLV1a88vljpLUggp-R4NFgNl_nDsh_jxbSb1avDRphkqA_Ri62dG0Mh8bP0BKYd3hicl6PA1QscO4EbSTWsX7qKrAQEmMMSaREdA5qdSo2xrl5A5dBnYjK3DUELK
955jf-cAxunBlerkrnrRSuJDwjOBErB7z2cHTwE0-84xelsBwod_g9M7LTfDS3G8LsBWNY',
    'cookies': {'nocookies': 'true'}
}
```

## Supported Captcha types

- `nocaptcha_proxyless`, `nocaptcha` : solving Google recaptcha with/without proxy
- `hcaptcha_proxyless`, `hcaptcha` : hCaptcha puzzle solving with/without proxy
- `funcaptcha_proxyless`, `funcaptcha` : solving FunCaptcha with/without proxy
- `geetest_proxyless`, `geetest` : GeeTest captcha recognition with/without proxy
- `recaptcha2_enterprise_proxyless`, `recaptcha2_enterprise` : solving Google reCAPTCHA Enterprise with/without proxy
- `recaptcha3_proxyless` : solving Google ReCaptcha v.3
- `turnstile_proxyless`, `turnstile` : solving Turnstile without proxy
- `image_to_text` : solve image captcha
- `compleximage_hcaptcha` : hCaptcha captcha solution
- `compleximage_recaptcha` : Google captcha solution
- `compleximage_funcaptcha` : Funcaptcha solving
