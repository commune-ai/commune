import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme } from '../context/ThemeContext';

const TaskCard = ({ task, completed, onPress }) => {
  const { theme } = useTheme();
  
  // Get priority color
  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high':
        return theme.error;
      case 'medium':
        return theme.warning;
      case 'low':
        return theme.success;
      default:
        return theme.primary;
    }
  };
  
  // Get difficulty icon
  const getDifficultyIcon = (difficulty) => {
    switch (difficulty) {
      case 'hard':
        return 'sword-cross';
      case 'medium':
        return 'sword';
      case 'easy':
        return 'shield-outline';
      default:
        return 'sword';
    }
  };
  
  return (
    <TouchableOpacity 
      style={[styles.container, { 
        backgroundColor: completed ? theme.secondaryBackground + '80' : theme.secondaryBackground,
        borderLeftColor: getPriorityColor(task.priority),
        opacity: completed ? 0.8 : 1,
      }]}
      onPress={onPress}
    >
      {/* Status Icon */}
      <View style={[styles.statusIcon, { 
        backgroundColor: completed ? theme.success : getPriorityColor(task.priority) 
      }]}>
        <Icon 
          name={completed ? 'check' : task.status === 'in_progress' ? 'play' : 'flag'} 
          size={16} 
          color="#fff" 
        />
      </View>
      
      {/* Task Content */}
      <View style={styles.content}>
        <Text style={[styles.title, { 
          color: theme.text,
          textDecorationLine: completed ? 'line-through' : 'none',
        }]}>
          {task.title}
        </Text>
        
        {task.description && !completed && (
          <Text style={[styles.description, { color: theme.text }]} numberOfLines={1}>
            {task.description}
          </Text>
        )}
        
        {/* Meta Info */}
        <View style={styles.meta}>
          {/* Due Date */}
          {task.dueDate && !completed && (
            <View style={styles.metaItem}>
              <Icon name="calendar" size={14} color={theme.primary} />
              <Text style={[styles.metaText, { color: theme.text }]}>
                {new Date(task.dueDate).toLocaleDateString()}
              </Text>
            </View>
          )}
          
          {/* Difficulty */}
          {task.difficulty && !completed && (
            <View style={styles.metaItem}>
              <Icon name={getDifficultyIcon(task.difficulty)} size={14} color={theme.primary} />
              <Text style={[styles.metaText, { color: theme.text }]}>
                {task.difficulty}
              </Text>
            </View>
          )}
          
          {/* Completed Date */}
          {completed && task.completedAt && (
            <View style={styles.metaItem}>
              <Icon name="check-circle" size={14} color={theme.success} />
              <Text style={[styles.metaText, { color: theme.text }]}>
                Completed {new Date(task.completedAt).toLocaleDateString()}
              </Text>
            </View>
          )}
        </View>
      </View>
      
      {/* Arrow Icon */}
      <Icon name="chevron-right" size={20} color={theme.text + '80'} />
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderLeftWidth: 4,
  },
  statusIcon: {
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  content: {
    flex: 1,
    marginRight: 8,
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  description: {
    fontSize: 14,
    marginBottom: 4,
    opacity: 0.7,
  },
  meta: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 12,
  },
  metaText: {
    fontSize: 12,
    marginLeft: 4,
    opacity: 0.8,
  },
});

export default TaskCard;
