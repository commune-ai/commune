import React from 'react';
import { View, StyleSheet } from 'react-native';

const ProgressBar = ({ progress, color, backgroundColor = '#E0E0E0', height = 6 }) => {
  // Ensure progress is between 0 and 100
  const clampedProgress = Math.min(100, Math.max(0, progress));
  
  return (
    <View style={[styles.container, { height, backgroundColor }]}>
      <View 
        style={[styles.progress, { 
          width: `${clampedProgress}%`, 
          backgroundColor: color,
          height,
        }]} 
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progress: {
    borderRadius: 3,
  },
});

export default ProgressBar;
