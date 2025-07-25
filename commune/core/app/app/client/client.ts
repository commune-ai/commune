
import config from '@/config.json';
import Key from '@/app/user/key';

export class Client {
  public url: string;

  /**
   * Initializes the Client instance with a base URL.
   * Ensures the URL starts with the specified mode (http/https).
   * @param url - The base URL for the client (default: value from config).
   * @param mode - The protocol mode ('http' or 'https', default: 'http').
   * @param key - An optional key for authentication or other purposes.
   */
  constructor(url: string = config.url, mode: string = 'http', key: Key) {
    if (!url.startsWith(`${mode}://`)) {
      url = `${mode}://${url}`;
    }
    this.url = url;
    this.key = key;
  }

  /**
   * Calls an API function with optional parameters and headers.
   * @param fn - The function name to be called (default: 'info').
   * @param params - The request parameters as a key-value object.
   * @param headers - Additional headers for the request.
   * @returns A promise resolving to the API response.
   */
  public async call(
    fn: string = 'info',
    params: Record<string, any> = {},
    headers: Record<string, string> = {}
  ): Promise<any> {
    try {
      return await this.async_call(fn, params);
    } catch (error) {
      console.error('Error in call method:', error);
      throw error;
    }
  }


  public sync_call(
    fn: string = 'info',
    params: Record<string, any> = {},
    headers: Record<string, string> = {}
  ): any {
    let future = this.async_call(fn, params);
    let result: any;
    future.then((res) => {
      result = res;
    }
    ).catch((error) => {
      console.error('Error in sync_call method:', error);
      throw error;
    }
    );
    while (result === undefined) {
      // Wait for the promise to resolve
      // This is a blocking call, use with caution
    }
    return result;
  }

  /**
   * Sends an asynchronous HTTP POST request to the specified function endpoint.
   * Supports both JSON and FormData payloads.
   * @param fn - The function name to be called.
   * @param params - The request parameters, either a JSON object or FormData.
   * @param headers - Additional headers for the request.
   * @returns A promise resolving to the API response.
   */
  private async async_call(
    fn: string = 'info',
    params: Record<string, any> | FormData = {},
  ): Promise<any> {
    let headers: Record<string, string> = {'Content-Type': 'application/json',   };
    let body: string | FormData;

    let timestamp = new Date().getTime() / 1000; // Current timestamp in seconds
    headers['time'] = timestamp.toString(); // Adds a timestamp to the headers
    
    console.log(`Calling function: ${fn} with params:`, headers.time);

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
        throw new Error(`HTTP error! status: ${response.status}`);
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

export default Client;
