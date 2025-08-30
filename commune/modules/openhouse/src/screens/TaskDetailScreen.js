import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme } from '../context/ThemeContext';
import { completeTaskAndUpdateStats, deleteTask, startTask, updateTask } from '../redux/actions/taskActions';
import ProgressBar from '../components/ProgressBar';

const TaskDetailScreen = ({ route, navigation }) => {
  const { taskId } = route.params;
  const { theme } = useTheme();
  const dispatch = useDispatch();
  const { tasks } = useSelector(state => state.tasks);
  const task = tasks.find(t => t.id === taskId);
  
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);
  
  if (!task) {
    return (
      <View style={[styles.container, { backgroundColor: theme.background }]}>
        <Text style={[styles.errorText, { color: theme.text }]}>Task not found</Text>
        <TouchableOpacity 
          style={[styles.button, { backgroundColor: theme.primary }]}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.buttonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }
  
  const handleStartTask = () => {
    dispatch(startTask(taskId));
    navigation.navigate('Focus', { taskId });
  };
  
  const handleCompleteTask = () => {
    dispatch(completeTaskAndUpdateStats(taskId));
    navigation.goBack();
  };
  
  const handleDeleteTask = () => {
    if (showConfirmDelete) {
      dispatch(deleteTask(taskId));
      navigation.goBack();
    } else {
      setShowConfirmDelete(true);
    }
  };
  
  const handleEditTask = () => {
    navigation.navigate('AddTask', { task });
  };
  
  const togglePriority = () => {
    const priorities = ['low', 'medium', 'high'];
    const currentIndex = priorities.indexOf(task.priority);
    const nextIndex = (currentIndex + 1) % priorities.length;
    
    dispatch(updateTask({
      ...task,
      priority: priorities[nextIndex],
    }));
  };
  
  // Calculate time spent
  const formatTimeSpent = () => {
    const totalSeconds = task.timeSpent || 0;
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };
  
  // Get color based on priority
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
  
  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.background }]}>
      {/* Task Header */}
      <View style={[styles.header, { backgroundColor: getPriorityColor(task.priority) }]}>
        <Text style={styles.headerTitle}>{task.title}</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity style={styles.headerButton} onPress={togglePriority}>
            <Icon name="flag" size={20} color="#fff" />
            <Text style={styles.headerButtonText}>{task.priority}</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.headerButton} onPress={handleEditTask}>
            <Icon name="pencil" size={20} color="#fff" />
          </TouchableOpacity>
        </View>
      </View>
      
      {/* Task Details */}
      <View style={styles.detailsContainer}>
        {/* Description */}
        <View style={[styles.section, { backgroundColor: theme.secondaryBackground }]}>
          <View style={styles.sectionHeader}>
            <Icon name="text" size={20} color={theme.primary} />
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Description</Text>
          </View>
          <Text style={[styles.description, { color: theme.text }]}>
            {task.description || 'No description provided'}
          </Text>
        </View>
        
        {/* Status & Progress */}
        <View style={[styles.section, { backgroundColor: theme.secondaryBackground }]}>
          <View style={styles.sectionHeader}>
            <Icon name="chart-timeline-variant" size={20} color={theme.primary} />
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Status & Progress</Text>
          </View>
          
          <View style={styles.statusRow}>
            <Text style={[styles.statusLabel, { color: theme.text }]}>Status:</Text>
            <View style={[styles.statusBadge, { backgroundColor: getStatusColor(task.status, theme) }]}>
              <Text style={styles.statusText}>{formatStatus(task.status)}</Text>
            </View>
          </View>
          
          {task.dueDate && (
            <View style={styles.statusRow}>
              <Text style={[styles.statusLabel, { color: theme.text }]}>Due Date:</Text>
              <Text style={[styles.statusValue, { color: theme.text }]}>
                {new Date(task.dueDate).toLocaleDateString()}
              </Text>
            </View>
          )}
          
          <View style={styles.statusRow}>
            <Text style={[styles.statusLabel, { color: theme.text }]}>Created:</Text>
            <Text style={[styles.statusValue, { color: theme.text }]}>
              {new Date(task.createdAt).toLocaleDateString()}
            </Text>
          </View>
          
          {task.completed && (
            <View style={styles.statusRow}>
              <Text style={[styles.statusLabel, { color: theme.text }]}>Completed:</Text>
              <Text style={[styles.statusValue, { color: theme.text }]}>
                {new Date(task.completedAt).toLocaleDateString()}
              </Text>
            </View>
          )}
          
          <View style={styles.statusRow}>
            <Text style={[styles.statusLabel, { color: theme.text }]}>Time Spent:</Text>
            <Text style={[styles.statusValue, { color: theme.text }]}>{formatTimeSpent()}</Text>
          </View>
          
          {task.estimatedTime && (
            <View style={styles.progressContainer}>
              <View style={styles.progressHeader}>
                <Text style={[styles.progressLabel, { color: theme.text }]}>Progress</Text>
                <Text style={[styles.progressValue, { color: theme.text }]}>
                  {formatTimeSpent()} / {task.estimatedTime}m
                </Text>
              </View>
              <ProgressBar 
                progress={Math.min(100, ((task.timeSpent || 0) / 60) / task.estimatedTime * 100)} 
                color={theme.primary} 
              />
            </View>
          )}
        </View>
        
        {/* Rewards */}
        <View style={[styles.section, { backgroundColor: theme.secondaryBackground }]}>
          <View style={styles.sectionHeader}>
            <Icon name="star" size={20} color={theme.primary} />
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Rewards</Text>
          </View>
          
          <View style={styles.rewardsContainer}>
            <View style={styles.rewardItem}>
              <Icon name="star" size={20} color={theme.accent} />
              <Text style={[styles.rewardText, { color: theme.text }]}>
                {calculateXP(task)} XP
              </Text>
            </View>
            
            <View style={styles.rewardItem}>
              <Icon name="coin" size={20} color={theme.accent} />
              <Text style={[styles.rewardText, { color: theme.text }]}>
                {calculatePoints(task)} Points
              </Text>
            </View>
          </View>
        </View>
      </View>
      
      {/* Action Buttons */}
      <View style={styles.actionsContainer}>
        {!task.completed && (
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: theme.primary }]}
            onPress={handleStartTask}
          >
            <Icon name="play" size={20} color="#fff" />
            <Text style={styles.actionButtonText}>Start Focus Session</Text>
          </TouchableOpacity>
        )}
        
        {!task.completed && (
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: theme.success }]}
            onPress={handleCompleteTask}
          >
            <Icon name="check" size={20} color="#fff" />
            <Text style={styles.actionButtonText}>Mark as Completed</Text>
          </TouchableOpacity>
        )}
        
        <TouchableOpacity 
          style={[styles.actionButton, { 
            backgroundColor: showConfirmDelete ? theme.error : theme.secondaryBackground,
            borderColor: showConfirmDelete ? theme.error : theme.border,
            borderWidth: 1,
          }]}
          onPress={handleDeleteTask}
        >
          <Icon 
            name={showConfirmDelete ? 'alert' : 'delete'} 
            size={20} 
            color={showConfirmDelete ? '#fff' : theme.error} 
          />
          <Text style={[styles.actionButtonText, { 
            color: showConfirmDelete ? '#fff' : theme.error 
          }]}>
            {showConfirmDelete ? 'Confirm Delete' : 'Delete Task'}
          </Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

// Helper functions
const formatStatus = (status) => {
  switch (status) {
    case 'pending':
      return 'To Do';
    case 'in_progress':
      return 'In Progress';
    case 'paused':
      return 'Paused';
    case 'completed':
      return 'Completed';
    default:
      return 'To Do';
  }
};

const getStatusColor = (status, theme) => {
  switch (status) {
    case 'pending':
      return theme.accent;
    case 'in_progress':
      return theme.primary;
    case 'paused':
      return theme.warning;
    case 'completed':
      return theme.success;
    default:
      return theme.accent;
  }
};

const calculateXP = (task) => {
  const priorityMultiplier = {
    high: 2,
    medium: 1.5,
    low: 1,
  };
  
  const difficultyMultiplier = {
    hard: 2,
    medium: 1.5,
    easy: 1,
  };
  
  const baseXP = 20;
  return Math.round(
    baseXP * 
    (priorityMultiplier[task.priority] || 1) * 
    (difficultyMultiplier[task.difficulty] || 1)
  );
};

const calculatePoints = (task) => {
  const priorityMultiplier = {
    high: 3,
    medium: 2,
    low: 1,
  };
  
  const basePoints = 10;
  return Math.round(basePoints * (priorityMultiplier[task.priority] || 1));
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    padding: 20,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 12,
  },
  headerActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  headerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 16,
  },
  headerButtonText: {
    color: '#fff',
    marginLeft: 4,
    fontWeight: 'bold',
    textTransform: 'capitalize',
  },
  detailsContainer: {
    padding: 16,
  },
  section: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  description: {
    fontSize: 14,
    lineHeight: 20,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  statusLabel: {
    fontSize: 14,
  },
  statusValue: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  statusBadge: {
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 12,
  },
  statusText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 12,
  },
  progressContainer: {
    marginTop: 8,
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  progressLabel: {
    fontSize: 14,
  },
  progressValue: {
    fontWeight: 'bold',
    fontSize: 14,
  },
  rewardsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  rewardItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.05)',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 16,
  },
  rewardText: {
    marginLeft: 8,
    fontWeight: 'bold',
  },
  actionsContainer: {
    padding: 16,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  actionButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    marginLeft: 8,
  },
  errorText: {
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 16,
  },
  button: {
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
});

export default TaskDetailScreen;
