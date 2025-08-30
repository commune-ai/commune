import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Animated, Vibration } from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme } from '../context/ThemeContext';
import { completeTaskAndUpdateStats, pauseTask } from '../redux/actions/taskActions';
import { addExperience, addPoints } from '../redux/actions/userActions';

const FocusScreen = ({ route, navigation }) => {
  const { taskId } = route.params;
  const { theme } = useTheme();
  const dispatch = useDispatch();
  const { tasks } = useSelector(state => state.tasks);
  const task = tasks.find(t => t.id === taskId);
  
  const [timeLeft, setTimeLeft] = useState(25 * 60); // 25 minutes in seconds
  const [isActive, setIsActive] = useState(false);
  const [isBreak, setIsBreak] = useState(false);
  const [breakCount, setBreakCount] = useState(0);
  const [completedIntervals, setCompletedIntervals] = useState(0);
  
  // Animation values
  const progressAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(1)).current;
  
  // Timer interval reference
  const timerRef = useRef(null);
  
  // Start the timer
  const startTimer = () => {
    setIsActive(true);
    
    // Pulse animation
    Animated.loop(
      Animated.sequence([
        Animated.timing(scaleAnim, {
          toValue: 1.05,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(scaleAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };
  
  // Pause the timer
  const pauseTimer = () => {
    setIsActive(false);
    scaleAnim.stopAnimation();
    scaleAnim.setValue(1);
  };
  
  // Reset the timer
  const resetTimer = () => {
    pauseTimer();
    setTimeLeft(isBreak ? (breakCount % 4 === 0 ? 15 * 60 : 5 * 60) : 25 * 60);
  };
  
  // Complete the current interval
  const completeInterval = () => {
    // Vibrate to notify completion
    Vibration.vibrate([0, 300, 100, 300]);
    
    if (isBreak) {
      // Break finished, start a new focus session
      setIsBreak(false);
      setTimeLeft(25 * 60);
    } else {
      // Focus session finished, start a break
      setCompletedIntervals(completedIntervals + 1);
      setIsBreak(true);
      
      // Every 4 intervals, take a longer break
      const nextBreakCount = breakCount + 1;
      setBreakCount(nextBreakCount);
      setTimeLeft(nextBreakCount % 4 === 0 ? 15 * 60 : 5 * 60);
      
      // Award points for completing a focus session
      dispatch(addExperience(15));
      dispatch(addPoints(10));
    }
    
    pauseTimer();
  };
  
  // Skip to the next interval
  const skipInterval = () => {
    if (isBreak) {
      setIsBreak(false);
      setTimeLeft(25 * 60);
    } else {
      setIsBreak(true);
      const nextBreakCount = breakCount + 1;
      setBreakCount(nextBreakCount);
      setTimeLeft(nextBreakCount % 4 === 0 ? 15 * 60 : 5 * 60);
    }
    pauseTimer();
  };
  
  // Format seconds to MM:SS
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  // Complete the task and return to home
  const finishTask = () => {
    dispatch(completeTaskAndUpdateStats(taskId));
    navigation.navigate('Home');
  };
  
  // Pause the task and return to home
  const exitFocus = () => {
    dispatch(pauseTask(taskId));
    navigation.goBack();
  };
  
  // Progress circle calculations
  const radius = 120;
  const circumference = 2 * Math.PI * radius;
  const totalTime = isBreak ? (breakCount % 4 === 0 ? 15 * 60 : 5 * 60) : 25 * 60;
  const progress = (totalTime - timeLeft) / totalTime;
  
  // Update the progress animation
  useEffect(() => {
    Animated.timing(progressAnim, {
      toValue: progress,
      duration: 300,
      useNativeDriver: false,
    }).start();
  }, [progress, progressAnim]);
  
  // Timer effect
  useEffect(() => {
    if (isActive) {
      timerRef.current = setInterval(() => {
        setTimeLeft(prev => {
          if (prev <= 1) {
            clearInterval(timerRef.current);
            completeInterval();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [isActive, isBreak]);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);
  
  // Calculate the stroke dashoffset based on progress
  const strokeDashoffset = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [circumference, 0],
  });
  
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
  
  return (
    <View style={[styles.container, { backgroundColor: theme.background }]}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={exitFocus}>
          <Icon name="arrow-left" size={24} color={theme.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: theme.text }]}>
          {isBreak ? 'Break Time' : 'Focus Mode'}
        </Text>
        <View style={{ width: 24 }} />
      </View>
      
      {/* Task Info */}
      <View style={[styles.taskInfo, { backgroundColor: theme.secondaryBackground }]}>
        <Text style={[styles.taskTitle, { color: theme.text }]}>{task.title}</Text>
        {task.description && (
          <Text style={[styles.taskDescription, { color: theme.text }]}>{task.description}</Text>
        )}
        <View style={styles.taskMeta}>
          <View style={[styles.priorityBadge, { backgroundColor: getPriorityColor(task.priority, theme) }]}>
            <Text style={styles.priorityText}>{task.priority}</Text>
          </View>
          <Text style={[styles.intervalText, { color: theme.text }]}>
            {completedIntervals} intervals completed
          </Text>
        </View>
      </View>
      
      {/* Timer Circle */}
      <View style={styles.timerContainer}>
        <Animated.View style={[styles.timerCircle, { transform: [{ scale: scaleAnim }] }]}>
          <Svg height={radius * 2 + 20} width={radius * 2 + 20} viewBox={`0 0 ${radius * 2 + 20} ${radius * 2 + 20}`}>
            {/* Background Circle */}
            <Circle
              cx={radius + 10}
              cy={radius + 10}
              r={radius}
              stroke={isBreak ? theme.secondary + '40' : theme.primary + '40'}
              strokeWidth={15}
              fill="transparent"
            />
            {/* Progress Circle */}
            <AnimatedCircle
              cx={radius + 10}
              cy={radius + 10}
              r={radius}
              stroke={isBreak ? theme.secondary : theme.primary}
              strokeWidth={15}
              fill="transparent"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              transform={`rotate(-90, ${radius + 10}, ${radius + 10})`}
            />
          </Svg>
          <View style={styles.timerTextContainer}>
            <Text style={[styles.timerText, { color: theme.text }]}>{formatTime(timeLeft)}</Text>
            <Text style={[styles.intervalLabel, { color: theme.text }]}>
              {isBreak ? (breakCount % 4 === 0 ? 'Long Break' : 'Short Break') : 'Focus Time'}
            </Text>
          </View>
        </Animated.View>
      </View>
      
      {/* Control Buttons */}
      <View style={styles.controlsContainer}>
        <TouchableOpacity 
          style={[styles.controlButton, { backgroundColor: theme.secondaryBackground }]}
          onPress={resetTimer}
        >
          <Icon name="refresh" size={24} color={theme.text} />
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.mainButton, { backgroundColor: isActive ? theme.error : theme.primary }]}
          onPress={isActive ? pauseTimer : startTimer}
        >
          <Icon name={isActive ? 'pause' : 'play'} size={32} color="#fff" />
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.controlButton, { backgroundColor: theme.secondaryBackground }]}
          onPress={skipInterval}
        >
          <Icon name="skip-next" size={24} color={theme.text} />
        </TouchableOpacity>
      </View>
      
      {/* Complete Task Button */}
      <TouchableOpacity 
        style={[styles.completeButton, { backgroundColor: theme.success }]}
        onPress={finishTask}
      >
        <Icon name="check" size={20} color="#fff" />
        <Text style={styles.completeButtonText}>Complete Task</Text>
      </TouchableOpacity>
      
      {/* Stats */}
      <View style={styles.statsContainer}>
        <View style={styles.statItem}>
          <Icon name="clock-outline" size={20} color={theme.primary} />
          <Text style={[styles.statText, { color: theme.text }]}>
            {completedIntervals * 25} minutes focused
          </Text>
        </View>
        <View style={styles.statItem}>
          <Icon name="coffee" size={20} color={theme.secondary} />
          <Text style={[styles.statText, { color: theme.text }]}>
            {breakCount} breaks taken
          </Text>
        </View>
      </View>
    </View>
  );
};

// Helper function to get color based on priority
const getPriorityColor = (priority, theme) => {
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

// Import SVG components for the timer circle
import Svg, { Circle } from 'react-native-svg';
const AnimatedCircle = Animated.createAnimatedComponent(Circle);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  taskInfo: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
  },
  taskTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  taskDescription: {
    fontSize: 14,
    marginBottom: 12,
    opacity: 0.8,
  },
  taskMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  priorityBadge: {
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 12,
  },
  priorityText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 12,
    textTransform: 'uppercase',
  },
  intervalText: {
    fontSize: 14,
  },
  timerContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 24,
  },
  timerCircle: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  timerTextContainer: {
    position: 'absolute',
    alignItems: 'center',
  },
  timerText: {
    fontSize: 48,
    fontWeight: 'bold',
  },
  intervalLabel: {
    fontSize: 16,
    opacity: 0.8,
  },
  controlsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  controlButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: 16,
  },
  mainButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    justifyContent: 'center',
    alignItems: 'center',
  },
  completeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  completeButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    marginLeft: 8,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 16,
  },
  statItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statText: {
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

export default FocusScreen;
