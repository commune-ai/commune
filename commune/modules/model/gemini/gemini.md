# Gemini Module

This module provides an interface for chatting with Gemini which is one of the most powerful Multimodal models.

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Description

Google has become the center of attention since the announcement of its new Generative AI family of models called the Gemini. As Google has stated, Google’s Gemini family of Large Language Models outperforms the existing State of The Art(SoTA) GPT model from OpenAI in more than 30+ benchmark tests. Not only in textual generations but Google with the Gemini Pro Vision model’s capabilities has proven that it has an upper edge on even the visual tasks compared to the GPT 4 Vision which recently went public from OpenAI. A few days ago, Google made two of its foundational models (Gemini-pro and Gemini-pro-vision) available to the public. In this module, we will create a multimodal chatbot with Gemini and Gradio.

## Installation

1. Copy the .env_copy to .env
2. Set up your environment variables in a `.env` file. A Google API key is required.
3. Run the following command: `pip install -r requirements.txt`

Google-generativeai: This Python library is for working with Google’s Gemini models. It provides functions to call the models like the gemini-pro and gemini-pro-vision
4. It must be run in the Google API supported regions.

## Usage

To run the module, use the following command:
While running the module, upload jpg image files.

```
c model.gemini gradio
```

One could be able to see  the following message in console:
  
  "Running on local URL:  http://127.0.0.1:7861
Running on public URL: https://4a258df9125674577d.gradio.live
  " 
  
Copy the public URL and open it in the browser.

## Contributing

We welcome contributions! Please feel free to submit a pull request or open an issue if you have suggestions or find a bug.

## License

This project is licensed under the MIT License - see the LICENSE file for details.