import { createHash } from 'crypto';
import Key from './key';

export interface AuthHeaders {
  data: string;
  time: string;
  key: string;
  signature: string;
  crypto_type?: string;
  hash_type?: string;
  data_hash?: string; // Optional, used for data verification
}

export class Auth {
  private key: Key;
  private hashType: string;
  private maxStaleness: number;
  private signatureKeys: string[];

  /**
   * Initialize the Auth class
   * @param key - The key to use for signing
   * @param hashType - The hash type to use for signing
   * @param maxStaleness - Maximum staleness allowed for timestamps (in seconds)
   * @param signatureKeys - The keys to use for signing
   */
  constructor(
    key: Key,
    hashType: string = 'sha256',
    maxStaleness: number = 60,
    signatureKeys: string[] = ['data', 'time']
  ) {
    this.key = key;
    this.hashType = hashType;
    this.maxStaleness = maxStaleness;
    this.signatureKeys = signatureKeys;
  }

  /**
   * Generate authentication headers with signature
   * @param data - The data to sign
   * @param key - Optional key override
   * @param crypto_type - Optional crypto type override
   * @returns Authentication headers with signature
   */
  public generate(data: any, key?: Key): AuthHeaders {
    const authKey = key || this.key;
    const headers: AuthHeaders = {
      data: this.hash(data),
      time: String(this.time()), // Unix timestamp in seconds
      key: authKey.address,
      signature: '',
      hash_type: this.hashType,
      crypto_type: authKey.crypto_type
    };

    
    // Create signature data object with only the specified keys
    let signatureData: Record<string, string> = {};
    this.signatureKeys.forEach(k => {
      if (k in headers) {
        signatureData[k] = headers[k as keyof AuthHeaders] as string;
      }
    });


    // Sign the data
    let signatureDataString = JSON.stringify(signatureData); // Ensure it's a plain object
    headers.signature = authKey.sign(signatureDataString)
    headers.data_hash = this.hash(headers.data); // Optional data hash for verification
      // Verify the signature
    const verified = this.key.verify( signatureDataString, headers.signature, headers.key);
    if (!verified) {
      throw new Error('Signature verification failed');
    }

    return headers;
  }

  /**
   * Alias for generate method
   */
  public headers(data: any, key?: Key): AuthHeaders {
    return this.generate(data, key);
  }

  /**
   * Verify and decode authentication headers
   * @param headers - The headers to verify
   * @param data - Optional data to verify against the hash
   * @returns The verified headers
   * @throws Error if verification fails
   */
  public time(): number {
    return Date.now() / 1000; // Returns current timestamp in seconds
  }

  public verify(headers: AuthHeaders, data?: any): AuthHeaders {
    // Check staleness
    const currentTime = this.time()
    const headerTime = parseFloat(headers.time);
    const staleness = Math.abs(currentTime - headerTime);
    
    if (staleness > this.maxStaleness) {
      throw new Error(`Token is stale: ${staleness}s > ${this.maxStaleness}s`);
    }

    if (!headers.signature) {
      throw new Error('Missing signature');
    }

    // if signature is 0x prefix string, remove it

    // Create signature data object for verification
    const signatureData: Record<string, string> = {};
    this.signatureKeys.forEach(k => {
      if (k in headers) {
        signatureData[k] = headers[k as keyof AuthHeaders] as string;
      }
    });

    const signatureDataString = JSON.stringify(signatureData); // Ensure it's a plain object


    let params = {
      message: signatureDataString,
      signature: headers.signature,
      public_key: headers.key,
    };
    console.log('params', params);

    let verified = this.key.verify(params.message, params.signature, params.public_key);

    // get boolean value of verified
    verified = Boolean(verified);

    // Verify data hash if provided
    if (data) {
      const rehashData = this.hash(data);
      if (headers.data !== rehashData) {
        throw new Error(`Invalid data hash: ${headers.data} !== ${rehashData}`);
      }
    }

    return verified;
  }

  /**
   * Alias for verify method
   */
  public verifyHeaders(headers: AuthHeaders, data?: any): AuthHeaders {
    return this.verify(headers, data);
  }

  /**
   * Hash the data using the specified hash type
   * @param data - The data to hash
   * @returns The hash string
   */
  private hash(data: any): string {


    
    let dataToHash: string;
    if (typeof data === 'string') {
      dataToHash = data;
    } else if (typeof data === 'object') {
      dataToHash = JSON.stringify(data);
    } else {
      dataToHash = String(data);
    }

    if (this.hashType == 'sha256') {
      return createHash('sha256').update(dataToHash).digest('hex');
    } else {
     throw new Error(`Invalid hash type: ${this.hashType}`);
    }

  }


  /**
   * Test the authentication flow
   * @param keyName - Name of the test key
   * @param crypto_type - Crypto type to test with
   * @returns Test results
   */
  public static async test(
    key: Key,
    crypto_type: string = 'sr25519'
  ): Promise<{ headers: AuthHeaders; verified: boolean }> {
    const data = { fn: 'test', params: { a: 1, b: 2 } };
    const auth = new Auth(key, crypto_type);
    
    // Generate headers
    const headers = auth.headers(data);
    
    // Verify headers without data
    auth.verify(headers);
    
    // Verify headers with data
    const verifiedHeaders = auth.verify(headers, data);
    
    return { 
      headers: verifiedHeaders,
      verified: true
    };
  }
}

export default Auth;
