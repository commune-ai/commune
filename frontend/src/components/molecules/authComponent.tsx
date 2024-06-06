import React, { useState } from 'react';

interface GeneratedKeys {
    gmailKey: string;
    metamaskKey: string;
    substrateKey: string;
    solanaKey: string;
}

const AuthComponent: React.FC = () => {
    const [userSeed, setUserSeed] = useState<string>('');
    const [isLoggedIn, setLoggedIn] = useState<boolean>(false);
    const [generatedKeys, setGeneratedKeys] = useState<GeneratedKeys>({
        gmailKey: '',
        metamaskKey: '',
        substrateKey: '',
        solanaKey: '',
    });

    const handleLogin = (method: string) => {
        console.log('----------------', method)
        // Perform authentication logic based on the selected method
        // Update the state with the generated keys and setLoggedIn(true)

        // Example: Generating keys based on a seed (use proper libraries for key generation)
        const gmailKey = generateKeyFromSeed(userSeed, 'gmail');
        const metamaskKey = generateKeyFromSeed(userSeed, 'metamask');
        const substrateKey = generateKeyFromSeed(userSeed, 'substrate');
        const solanaKey = generateKeyFromSeed(userSeed, 'solana');

        setGeneratedKeys({
            gmailKey,
            metamaskKey,
            substrateKey,
            solanaKey,
        });

        setLoggedIn(true);
    };

    const generateKeyFromSeed = (seed: string, method: string): string => {
        console.log('------------------', seed, method)
        // Implement key generation logic based on the seed and method
        // Use appropriate libraries for key generation for each blockchain
        // Ethereum (EVM) key generation using ethers.js
        // const wallet = Wallet.fromMnemonic(seed);
        // const ethAddress = wallet.address;

        return 'ethAddress';

    };

    return (
        <div>
            {!isLoggedIn ? (
                <div>
                    <label>
                        Enter Seed:
                        <input
                            type="text"
                            value={userSeed}
                            onChange={({ target: { value } }) => setUserSeed(value)}
                        />
                    </label>
                    <button onClick={() => handleLogin('gmail')}>Login with Gmail</button>
                    <button onClick={() => handleLogin('metamask')}>Login with MetaMask</button>
                    <button onClick={() => handleLogin('substrate')}>Login with Substrate</button>
                    <button onClick={() => handleLogin('solana')}>Login with Solana</button>
                </div>
            ) : (
                <div>
                    <p>Successfully Logged In!</p>
                    <p>Gmail Key: {generatedKeys.gmailKey}</p>
                    <p>MetaMask Key: {generatedKeys.metamaskKey}</p>
                    <p>Substrate Key: {generatedKeys.substrateKey}</p>
                    <p>Solana Key: {generatedKeys.solanaKey}</p>
                </div>
            )}
        </div>
    );
};

export default AuthComponent;
