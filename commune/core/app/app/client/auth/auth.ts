import { createHash } from 'crypto';
import Key from '@/app/user/key';

export interface AuthHeaders {
  data: string;
  time: string;
  key: string;
  signature: string;
  crypto_type?: string;
}

export class Auth {
  private key: Key;
  private cryptoType: string;
  private hashType: string;
  private maxStaleness: number;
  private signatureKeys: string[];

  /**
   * Initialize the Auth class
   * @param key - The key to use for signing
   * @param cryptoType - The crypto type to use for signing
   * @param hashType - The hash type to use for signing
   * @param maxStaleness - Maximum staleness allowed for timestamps (in seconds)
   * @param signatureKeys - The keys to use for signing
   */
  constructor(
    key: Key,
    cryptoType: string = 'sr25519',
    hashType: string = 'sha256',
    maxStaleness: number = 60,
    signatureKeys: string[] = ['data', 'time']
  ) {
    this.key = key;
    this.cryptoType = cryptoType;
    this.hashType = hashType;
    this.maxStaleness = maxStaleness;
    this.signatureKeys = signatureKeys;
  }

  /**
   * Generate authentication headers with signature
   * @param data - The data to sign
   * @param key - Optional key override
   * @param cryptoType - Optional crypto type override
   * @returns Authentication headers with signature
   */
  public generate(data: any, key?: Key, cryptoType?: string): AuthHeaders {
    const authKey = key || this.key;
    const authCryptoType = cryptoType || this.cryptoType;

    const result: AuthHeaders = {
      data: this.hash(data),
      time: String(Date.now() / 1000), // Unix timestamp in seconds
      key: authKey.address,
      signature: ''
    };

    // Create signature data object with only the specified keys
    const signatureData: Record<string, string> = {};
    this.signatureKeys.forEach(k => {
      if (k in result) {
        signatureData[k] = result[k as keyof AuthHeaders] as string;
      }
    });

    // Sign the data
    result.signature = authKey.sign(signatureData, 'string');
    
    if (authCryptoType !== this.cryptoType) {
      result.crypto_type = authCryptoType;
    }

    return result;
  }

  /**
   * Alias for generate method
   */
  public headers(data: any, key?: Key, cryptoType?: string): AuthHeaders {
    return this.generate(data, key, cryptoType);
  }

  /**
   * Verify and decode authentication headers
   * @param headers - The headers to verify
   * @param data - Optional data to verify against the hash
   * @returns The verified headers
   * @throws Error if verification fails
   */
  public verify(headers: AuthHeaders, data?: any): AuthHeaders {
    // Check staleness
    const currentTime = Date.now() / 1000;
    const headerTime = parseFloat(headers.time);
    const staleness = Math.abs(currentTime - headerTime);
    
    if (staleness > this.maxStaleness) {
      throw new Error(`Token is stale: ${staleness}s > ${this.maxStaleness}s`);
    }

    if (!headers.signature) {
      throw new Error('Missing signature');
    }

    // Create signature data object for verification
    const signatureData: Record<string, string> = {};
    this.signatureKeys.forEach(k => {
      if (k in headers) {
        signatureData[k] = headers[k as keyof AuthHeaders] as string;
      }
    });

    // Verify the signature
    const cryptoType = headers.crypto_type || this.cryptoType;
    const verified = this.key.verify(
      signatureData,
      headers.signature,
      headers.key,
      cryptoType
    );

    if (!verified) {
      throw new Error('Invalid signature');
    }

    // Verify data hash if provided
    if (data !== undefined) {
      const rehashData = this.hash(data);
      if (headers.data !== rehashData) {
        throw new Error(`Invalid data hash: ${headers.data} !== ${rehashData}`);
      }
    }

    return headers;
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
    if (this.hashType !== 'sha256') {
      throw new Error(`Invalid hash type: ${this.hashType}`);
    }

    let dataToHash: string;
    if (typeof data === 'string') {
      dataToHash = data;
    } else if (typeof data === 'object') {
      dataToHash = JSON.stringify(data);
    } else {
      dataToHash = String(data);
    }

    return createHash('sha256').update(dataToHash).digest('hex');
  }

  /**
   * Get the crypto type with fallback to instance default
   * @param cryptoType - Optional crypto type override
   * @returns The crypto type to use
   */
  private getCryptoType(cryptoType?: string): string {
    const type = cryptoType || this.cryptoType;
    if (!['sr25519', 'ed25519'].includes(type)) {
      throw new Error(`Invalid crypto type: ${type}`);
    }
    return type;
  }

  /**
   * Get the key with fallback to instance default
   * @param key - Optional key override
   * @param cryptoType - Optional crypto type for key creation
   * @returns The key to use
   */
  private getKey(key?: Key | string, cryptoType?: string): Key {
    const authCryptoType = this.getCryptoType(cryptoType);
    
    if (!key) {
      return this.key;
    }

    if (typeof key === 'string') {
      // Assuming Key has a static method to get key by name
      // This would need to be implemented in the Key class
      throw new Error('String key lookup not implemented');
    }

    if (!key.address) {
      throw new Error(`Invalid key: missing address`);
    }

    return key;
  }

  /**
   * Test the authentication flow
   * @param keyName - Name of the test key
   * @param cryptoType - Crypto type to test with
   * @returns Test results
   */
  public static async test(
    key: Key,
    cryptoType: string = 'sr25519'
  ): Promise<{ headers: AuthHeaders; verified: boolean }> {
    const data = { fn: 'test', params: { a: 1, b: 2 } };
    const auth = new Auth(key, cryptoType);
    
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
