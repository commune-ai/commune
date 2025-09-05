
import {Auth, AuthHeaders} from '@/app/client/auth';
import Key from '@/app/key';


// load config from environment variables or default values
// get the process without using nodejs process

export class Client {
  public url: string;

  /**
   * Initializes the Client instance with a base URL.
   * Ensures the URL starts with the specified mode (http/https).
   * @param url - The base URL for the client (default: value from config).
   * @param mode - The protocol mode ('http' or 'https', default: 'http').
   * @param key - An optional key for authentication or other purposes.
   */
  constructor(url?: string , key: Key, mode: string = 'http' ) {

    this.url = this.getUrl(url);
    this.key = key;
    this.auth = new Auth(key);

  }

  public getUrl(url: string, mode: string = 'http'): string {

    url = url || process.env.NEXT_PUBLIC_API_URL || 'localhost:8000';
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
  private async call(fn: string = 'info',params: Record<string, any> | FormData = {}, cost = 0, headers = {}): Promise<any> {
    let body: string | FormData;


    let start_time = Date.now();
    headers = this.auth.generate({'fn': fn, 'params': params}, cost);
    console.log(`Calling function: ${fn} with params:`, params);
    console.log(`headers`,  headers)
    body = JSON.stringify(params);

    
    headers['Content-Type'] = 'application/json'; // Set Content-Type for JSON payload
    const url: string = `${this.url}/${fn}`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: body,
      });

        let delta_seconds = (Date.now() - start_time) / 1000;

        console.log(`response ${this.url}/${fn} generation took ${delta_seconds} seconds`);

      
      if (!response.ok) {
        // Handle HTTP errors
        // if success field exists and is false, return the error message

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
        let result =  await response.json();
        if (result && result.success === false) {
          let error_msg = JSON.stringify(result);
          throw new Error(`API Error: ${error_msg}`);
        }
        return result;
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

