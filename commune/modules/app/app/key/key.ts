import {
    blake2AsHex,
    sr25519PairFromSeed,
    encodeAddress,
    decodeAddress,
    sr25519Sign,
    sr25519Verify,
  } from '@polkadot/util-crypto'
  import { hexToU8a, u8aToHex } from '@polkadot/util'
  import { secp256k1 } from '@noble/curves/secp256k1'
  
  // Define the structure of a key object
  export interface WalletType {
    address: string
    crypto_type: 'sr25519' | 'ecdsa'
    public_key: string
    private_key: string
  }
  
  // Define allowed signature types
  type signature_t = 'sr25519' | 'ecdsa'
  
  export class Key {
    private private_key: string // Stores the private key of the key
    public public_key: string // Stores the public key of the key
    public address: string // Stores the key's address
    public crypto_type: signature_t // Defines the signature type used by the key
  
    /**
     * Constructs a new Key instance using a password as the seed.
     * @param password - The password used to generate keys.
     * @param crypto_type - The cryptographic algorithm (default: 'sr25519').
     */
    constructor(password : string, crypto_type: signature_t = 'sr25519') {
      const { public_key, private_key, address } = this.fromPassword(
        password,
        crypto_type
      )
      this.public_key = public_key
      this.private_key = private_key
      this.address = address
      this.crypto_type = crypto_type
    }
  
    /**
     * Generates a key from a password.
     * @param password - The password used to derive the keypair.
     * @param crypto_type - The cryptographic algorithm (default: 'sr25519').
     * @returns A WalletType object containing address, public_key, and private_key.
     */


    private fromPassword(
      password: string,
      crypto_type: signature_t = 'sr25519'
    ): WalletType {
      if (!password || typeof password !== 'string') {
        throw new Error('Invalid password provided')
      }
  
      // Derive a seed from the password using Blake2 hashing
      const seedHex = blake2AsHex(password, 256)
      const seedBytes = hexToU8a(seedHex)
      let key: WalletType
  
      if (crypto_type === 'sr25519') {
        // Generate sr25519 keypair - sr25519PairFromSeed is synchronous
        const keyPair = sr25519PairFromSeed(seedBytes)
        const address = encodeAddress(keyPair.publicKey, 42)
  
        key = {
          address,
          crypto_type: 'sr25519',
          public_key: u8aToHex(keyPair.publicKey),
          private_key: u8aToHex(keyPair.secretKey),
        }
      } else if (crypto_type === 'ecdsa') {
          // Generate ECDSA keypair using secp256k1
          const public_key = secp256k1.getPublicKey(seedHex)
    
          key = {
            address: u8aToHex(public_key),
            crypto_type: 'ecdsa',
            public_key : u8aToHex(public_key),
            private_key: seedHex,
          }
      } else {
        throw new Error('Unsupported crypto type')
      }
  
      return key
    }
  
    /**
     * Signs a message using the key's private key.
     * @param message - The message to sign.
     * @returns A signature string in hex format.
     */

    private resolvePublicKey(public_key: string): string {
      // Check if the public key is in SS58 format and convert it to hex if necessary
      if ((public_key.startsWith('5') || public_key.startsWith('H')) && public_key.length === 48) {
        // Convert SS58 address to public key bytes
        const publicKeyBytes = decodeAddress(public_key, false, 42);
        return u8aToHex(publicKeyBytes);
      } else if (public_key.startsWith('0x')) {
        // If the public key is already in hex format, return it as is
        return public_key;
      } else {
        throw new Error('Invalid public key format')
      }
    }

    public sign(message: string): Promise<string> {
      if (!message) {
        throw new Error('Empty message cannot be signed')
      }

      const messageBytes = new TextEncoder().encode(message)
  
      if (this.crypto_type === 'sr25519') {
        const signature = sr25519Sign(messageBytes, {
          publicKey: hexToU8a(this.public_key),
          secretKey: hexToU8a(this.private_key),
        })
        return u8aToHex(signature)
      } else if (this.crypto_type === 'ecdsa') {
          const messageHash = blake2AsHex(message)
          const signature = secp256k1.sign(
            hexToU8a(messageHash),
            hexToU8a(this.private_key)
          ).toDERRawBytes()
          return u8aToHex(signature)
      } else {
        throw new Error('Unsupported crypto type')
      }
    }
  
    /**
     * Verifies a signature against a message and public key.
     * @param message - The original message.
     * @param signature - The signature to verify (in hex format).
     * @param public_key - The public key corresponding to the private key used for signing (in hex format).
     * @returns A boolean indicating whether the signature is valid.
     */
    public verify(
      message: string,
      signature: string,
      public_key: string
    ): boolean {
      if (!message || !signature || !public_key) {
        throw new Error(`Invalid verification `)
      }
      
      const sigType = this.crypto_type


      const messageBytes = new TextEncoder().encode(message)

      public_key = this.resolvePublicKey(public_key)
      
      if (sigType === 'sr25519') {
        return sr25519Verify(
          messageBytes,
          hexToU8a(signature),
          hexToU8a(public_key)
        )
      } else if (sigType === 'ecdsa') {
        const messageHash = blake2AsHex(message)
        return secp256k1.verify(
          hexToU8a(signature),
          hexToU8a(messageHash),
          hexToU8a(public_key)
        )
      } else {
        throw new Error('Unsupported crypto type')
      }
    }
  
    /**
     * Encodes a string message into a Uint8Array.
     * @param message - The message to encode.
     * @returns A Uint8Array representation of the message.
     */
    encode(message: string): Uint8Array {
      return new TextEncoder().encode(message)
    }
  
    /**
     * Decodes a Uint8Array back into a string.
     * @param message - The Uint8Array to decode.
     * @returns A string representation of the message.
     */
    decode(message: Uint8Array): string {
      return new TextDecoder().decode(message)
    }
  }