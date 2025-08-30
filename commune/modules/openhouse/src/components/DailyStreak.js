import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme } from '../context/ThemeContext';

const DailyStreak = ({ streak }) => {
  const { theme } = useTheme();
  
  // Get streak message based on streak count
  const getStreakMessage = () => {
    if (streak === 0) return "Start your streak today!";
    if (streak === 1) return "First day of your streak!";
    if (streak < 3) return `${streak} day streak! Keep it up!`;
    if (streak < 7) return `${streak} day streak! You're on fire!`;
    if (streak < 14) return `${streak} day streak! Incredible!`;
    if (streak < 30) return `${streak} day streak! Unstoppable!`;
    return `${streak} day streak! Legendary!`;
  };
  
  // Get flame color based on streak
  const getFlameColor = () => {
    if (streak < 3) return theme.primary;
    if (streak < 7) return theme.accent;
    if (streak < 14) return '#FF9800'; // Orange
    if (streak < 30) return '#FF5722'; // Deep Orange
    return '#F44336'; // Red
  };
  
  return (
    <View style={[styles.container, { backgroundColor: theme.secondaryBackground }]}>
      <View style={[styles.flameContainer, { backgroundColor: getFlameColor() }]}>
        <Icon name="fire" size={24} color="#fff" />
      </View>
      <View style={styles.textContainer}>
        <Text style={[styles.streakCount, { color: theme.text }]}>{streak} Day Streak</Text>
        <Text style={[styles.streakMessage, { color: theme.text }]}>{getStreakMessage()}</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 12,
    marginBottom: 16,
  },
  flameContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  textContainer: {
    flex: 1,
  },
  streakCount: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 2,
  },
  streakMessage: {
    fontSize: 14,
    opacity: 0.8,
  },
});

export default DailyStreak;
