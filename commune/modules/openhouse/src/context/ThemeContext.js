import React, { createContext, useState, useContext, useEffect } from 'react';
import { Appearance } from 'react-native';

const ThemeContext = createContext();

const themes = {
  light: {
    background: '#FFFFFF',
    secondaryBackground: '#F5F5F5',
    text: '#333333',
    primary: '#6200EE',    // Purple for primary actions
    secondary: '#03DAC6',  // Teal for secondary elements
    accent: '#FF9800',     // Orange for accents and highlights
    success: '#4CAF50',    // Green for success states
    warning: '#FFC107',    // Amber for warnings
    error: '#F44336',      // Red for errors
    card: '#FFFFFF',
    border: '#E0E0E0',
    notification: '#FF9800',
  },
  dark: {
    background: '#121212',
    secondaryBackground: '#1E1E1E',
    text: '#E0E0E0',
    primary: '#BB86FC',    // Light purple for primary actions
    secondary: '#03DAC6',   // Teal for secondary elements
    accent: '#FF9800',      // Orange for accents and highlights
    success: '#4CAF50',     // Green for success states
    warning: '#FFC107',     // Amber for warnings
    error: '#F44336',       // Red for errors
    card: '#1E1E1E',
    border: '#333333',
    notification: '#FF9800',
  },
  adhd: {
    // High contrast, ADHD-friendly theme with reduced visual noise
    background: '#FFFFFF',
    secondaryBackground: '#F0F0F0',
    text: '#000000',          // Black text for high contrast
    primary: '#7B1FA2',       // Deep purple for primary actions
    secondary: '#00796B',     // Deep teal for secondary elements
    accent: '#FF6D00',        // Deep orange for accents
    success: '#2E7D32',       // Deep green for success
    warning: '#FF8F00',       // Deep amber for warnings
    error: '#C62828',         // Deep red for errors
    card: '#FFFFFF',
    border: '#BDBDBD',
    notification: '#FF6D00',
  }
};

export const ThemeProvider = ({ children }) => {
  const deviceColorScheme = Appearance.getColorScheme();
  const [themeType, setThemeType] = useState('adhd'); // Default to ADHD-friendly theme

  useEffect(() => {
    // Listen for device theme changes
    const subscription = Appearance.addChangeListener(({ colorScheme }) => {
      if (themeType === 'light' || themeType === 'dark') {
        setThemeType(colorScheme);
      }
    });

    return () => subscription.remove();
  }, [themeType]);

  const toggleTheme = (theme) => {
    setThemeType(theme);
  };

  const theme = themes[themeType];

  return (
    <ThemeContext.Provider value={{ theme, themeType, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);
