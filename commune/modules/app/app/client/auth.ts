import { createHash } from 'crypto';
import Key from '@/app/key';

export interface AuthHeaders {
  data: string;
  time: string;
  key: string;
  signature: string;
  crypto_type?: string;
  hash_type?: string;
  cost?: string;
  sigData?: string; // Added sigData field
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
    signatureKeys: string[] = ['data', 'time', 'cost']
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
  public generate(data: any, cost: number = 10): AuthHeaders {


    const headers: AuthHeaders = {
      data: this.hash(data),
      time: String(this.time()), // Unix timestamp in seconds
      key: this.key.address,
      signature: '',
      hash_type: this.hashType,
      crypto_type: this.key.crypto_type,
      cost: String(cost), 
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

    headers.sigData = signatureDataString;
    headers.signature = this.key.sign(signatureDataString)
      // Verify the signature
    const verified = this.key.verify( signatureDataString, headers.signature, headers.key);
    if (!verified) {
      throw new Error('Signature verification failed');
    }

    return headers;
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

    if (this.hashType == 'sha256') {
      let dataToHash = JSON.stringify(data);
      let hash =  createHash('sha256').update(dataToHash).digest('hex');
       return hash;
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
