
import config from '@/config.json';
import {Key, Auth, AuthHeaders} from '@/app/key';

export class Client {
  public url: string;

  /**
   * Initializes the Client instance with a base URL.
   * Ensures the URL starts with the specified mode (http/https).
   * @param url - The base URL for the client (default: value from config).
   * @param mode - The protocol mode ('http' or 'https', default: 'http').
   * @param key - An optional key for authentication or other purposes.
   */
  constructor(url?: string , key?: Key, mode: string = 'http' ) {

    this.url = this.getUrl(url);
    this.key = key;
    this.auth = new Auth(key);

    console.log(`Client initialized with URL: ${this.url} and key: ${this.key ? this.key.public_key : 'No key provided'}`);


  }


  public getUrl(url: string, mode: string = 'http'): string {
    url = url || '0.0.0.0:8000';
    if (!url.startsWith(mode + '://')) {
      url = mode + '://' + url ;
    }
    return url;
  }


  /**
   * Sends an asynchronous HTTP POST request to the specified function endpoint.
   * Supports both JSON and FormData payloads.
   * @param fn - The function name to be called.
   * @param params - The request parameters, either a JSON object or FormData.
   * @param headers - Additional headers for the request.
   * @returns A promise resolving to the API response.
   */
  private async call(
    fn: string = 'info',
    params: Record<string, any> | FormData = {},
    headers: Record<string, string> = {}
  ): Promise<any> {
    let body: string | FormData;

    let timestamp = new Date().getTime() / 1000; // Current timestamp in seconds
    headers['time'] = timestamp.toString(); // Adds a timestamp to the headers
    
    console.log(`Calling function: ${fn} with params:`, params);

    if (params instanceof FormData) {
      body = params; // FormData should not have Content-Type manually set
    } else {
      body = JSON.stringify(params);
      headers['Content-Type'] = '';
    }
    
    const url: string = `${this.url}/${fn}`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: body,
      });

      if (!response.ok) {
        // Handle HTTP errors
        if (response.status === 401) {
          throw new Error('Unauthorized access - please check your authentication credentials.');
        } else if (response.status === 404) {
          throw new Error('Resource not found - please check the URL or function name.');
        } else if (response.status === 500) {
          throw new Error('Internal server error - please try again later.');
        } else {
          throw new Error(`Unexpected error - status code: ${response.status}`);
        }
      }

      const contentType = response.headers.get('Content-Type');
      if (contentType?.includes('text/event-stream')) {
        return this.handleStream(response);
      }
      if (contentType?.includes('application/json')) {
        return await response.json();
      }
      return await response.text();
    } catch (error) {
      console.error('Request failed:', error);
      return { error: (error as Error).message };
    }
  }

  /**
   * Handles streaming responses for server-sent events (SSE).
   * Reads the stream in chunks and processes incoming data.
   * @param response - The fetch Response object.
   * @returns A promise that resolves when the stream is fully read.
   */
  private async handleStream(response: Response): Promise<void> {
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      console.log(decoder.decode(value)); // Process streaming data as needed
    }
  }


}

