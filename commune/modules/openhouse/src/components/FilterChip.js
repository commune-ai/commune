import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';
import { useTheme } from '../context/ThemeContext';

const FilterChip = ({ label, selected, onPress, color }) => {
  const { theme } = useTheme();
  
  return (
    <TouchableOpacity 
      style={[styles.chip, { 
        backgroundColor: selected ? (color || theme.primary) : theme.secondaryBackground,
        borderColor: color || theme.primary,
        borderWidth: 1,
      }]}
      onPress={onPress}
    >
      <Text style={[styles.label, { 
        color: selected ? '#fff' : color || theme.primary 
      }]}>{label}</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  chip: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 16,
    marginRight: 8,
    marginBottom: 8,
  },
  label: {
    fontWeight: 'bold',
    fontSize: 12,
  },
});

export default FilterChip;
