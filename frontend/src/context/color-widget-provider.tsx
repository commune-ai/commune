import React, { createContext, useState, useContext, ReactNode } from 'react';

// Define the type for the color context
type ColorContextType = {
    color: string;
    changeColor: (newColor: string) => void;
};

// Create a context to manage the color state
const ColorContext = createContext<ColorContextType | undefined>(undefined);

// Custom hook to access the color context
export const useColorContext = (): ColorContextType => {
    const context = useContext(ColorContext);
    if (!context) {
        throw new Error('useColorContext must be used within a ColorProvider');
    }
    return context;
};

// Color provider component to wrap around the entire application
type ColorProviderProps = {
    children: ReactNode;
};

export const ColorProvider: React.FC<ColorProviderProps> = ({ children }) => {
    const [color, setColor] = useState<string>('#161C3B'); // Initial color state

    const changeColor = (newColor: string) => {
        setColor(newColor);
    };

    const contextValue: ColorContextType = {
        color,
        changeColor,
    };

    return <ColorContext.Provider value={contextValue}>{children}</ColorContext.Provider>;
};
