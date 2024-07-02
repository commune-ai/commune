# Web Surfing Agent

The Web Surfing Agent is a sophisticated software module built on the langchain framework. It's designed to perform automated web searches and content summarization.

## Description

Leveraging the DuckDuckGoSearchResults tool and the ChatOpenAI model, this agent can fetch, summarize, and present information from the web in a concise format. It's equipped with a custom web fetching tool that uses requests and BeautifulSoup to parse HTML content, and a summarization tool that employs a ChatOpenAI model to distill the web page's content.

## Features

- **Automated web searches**: Uses DuckDuckGo for web searches.
- **Custom headers**: Fetches web page content with custom headers for better request handling.
- **HTML Parsing**: Parses HTML to text for easy summarization.
- **Content Summarization**: Summarizes content using GPT-3.5 Turbo via the langchain LLMChain.
- **Integration**: Easy integration with other langchain modules for extended functionality.

## Installation

To install, set up your environment variables in a .env file. An OpenAI key is required.

## Usage

Test the module with a custom query:

```
c model.langchain_agent search_the_web "what is the main idea of the book 'Origin'?"
```

```
c model.langchain_agent gradio
```
## Contributing

We welcome contributions! Please feel free to submit a pull request or open an issue if you have suggestions or find a bug.

## License

This project is licensed under the MIT License - see the LICENSE file for details.