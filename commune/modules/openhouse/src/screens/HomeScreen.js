import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image } from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme } from '../context/ThemeContext';
import TaskCard from '../components/TaskCard';
import ProgressBar from '../components/ProgressBar';
import DailyStreak from '../components/DailyStreak';
import { checkDailyStreak } from '../redux/actions/userActions';
import { initializeDefaultAchievements } from '../redux/actions/achievementActions';
import { initializeDefaultRewards } from '../redux/actions/rewardActions';

const HomeScreen = ({ navigation }) => {
  const { theme } = useTheme();
  const dispatch = useDispatch();
  const { tasks, activeTask } = useSelector(state => state.tasks);
  const { stats } = useSelector(state => state.user);
  const { unlockedAchievements } = useSelector(state => state.achievements);
  
  // Initialize default data
  useEffect(() => {
    dispatch(initializeDefaultAchievements());
    dispatch(initializeDefaultRewards());
    dispatch(checkDailyStreak());
  }, [dispatch]);
  
  // Filter tasks for today's view
  const todayTasks = tasks.filter(task => 
    !task.completed && 
    (task.priority === 'high' || task.dueDate === new Date().toISOString().split('T')[0])
  ).slice(0, 3);
  
  // Get recently completed tasks
  const recentlyCompleted = tasks
    .filter(task => task.completed)
    .sort((a, b) => new Date(b.completedAt) - new Date(a.completedAt))
    .slice(0, 2);
  
  // Calculate progress percentage for level
  const levelProgress = (stats.experience / stats.experienceToNextLevel) * 100;
  
  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.background }]}>
      {/* Header Section with User Stats */}
      <View style={styles.header}>
        <View style={styles.userInfo}>
          <View style={[styles.avatar, { backgroundColor: theme.primary }]}>
            <Text style={styles.avatarText}>{stats.level}</Text>
          </View>
          <View style={styles.userStats}>
            <Text style={[styles.username, { color: theme.text }]}>Hero Level {stats.level}</Text>
            <View style={styles.progressContainer}>
              <ProgressBar progress={levelProgress} color={theme.primary} />
              <Text style={[styles.progressText, { color: theme.text }]}>
                {stats.experience}/{stats.experienceToNextLevel} XP
              </Text>
            </View>
          </View>
        </View>
        <TouchableOpacity 
          style={[styles.pointsContainer, { backgroundColor: theme.accent }]}
          onPress={() => navigation.navigate('Rewards')}
        >
          <Icon name="coin" size={16} color="#fff" />
          <Text style={styles.pointsText}>{stats.points}</Text>
        </TouchableOpacity>
      </View>
      
      {/* Streak Section */}
      <DailyStreak streak={stats.streak} />
      
      {/* Active Quest Section */}
      {activeTask && (
        <View style={[styles.section, { backgroundColor: theme.secondaryBackground }]}>
          <View style={styles.sectionHeader}>
            <Icon name="sword" size={20} color={theme.primary} />
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Active Quest</Text>
          </View>
          <TaskCard 
            task={activeTask} 
            onPress={() => navigation.navigate('TaskDetail', { taskId: activeTask.id })} 
          />
          <TouchableOpacity 
            style={[styles.focusButton, { backgroundColor: theme.primary }]}
            onPress={() => navigation.navigate('Focus', { taskId: activeTask.id })}
          >
            <Icon name="timer" size={16} color="#fff" />
            <Text style={styles.focusButtonText}>Enter Focus Mode</Text>
          </TouchableOpacity>
        </View>
      )}
      
      {/* Today's Quests Section */}
      <View style={[styles.section, { backgroundColor: theme.secondaryBackground }]}>
        <View style={styles.sectionHeader}>
          <Icon name="calendar-today" size={20} color={theme.primary} />
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Today's Quests</Text>
        </View>
        
        {todayTasks.length > 0 ? (
          todayTasks.map(task => (
            <TaskCard 
              key={task.id} 
              task={task} 
              onPress={() => navigation.navigate('TaskDetail', { taskId: task.id })} 
            />
          ))
        ) : (
          <View style={styles.emptyState}>
            <Icon name="check-circle-outline" size={40} color={theme.success} />
            <Text style={[styles.emptyStateText, { color: theme.text }]}>
              No high priority tasks for today!
            </Text>
          </View>
        )}
        
        <TouchableOpacity 
          style={[styles.addButton, { backgroundColor: theme.secondary }]}
          onPress={() => navigation.navigate('AddTask')}
        >
          <Icon name="plus" size={16} color="#fff" />
          <Text style={styles.addButtonText}>Add New Quest</Text>
        </TouchableOpacity>
      </View>
      
      {/* Recently Completed Section */}
      {recentlyCompleted.length > 0 && (
        <View style={[styles.section, { backgroundColor: theme.secondaryBackground }]}>
          <View style={styles.sectionHeader}>
            <Icon name="check-circle" size={20} color={theme.primary} />
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Recently Completed</Text>
          </View>
          
          {recentlyCompleted.map(task => (
            <TaskCard 
              key={task.id} 
              task={task} 
              completed 
              onPress={() => navigation.navigate('TaskDetail', { taskId: task.id })} 
            />
          ))}
        </View>
      )}
      
      {/* Recent Achievements Section */}
      {unlockedAchievements.length > 0 && (
        <View style={[styles.section, { backgroundColor: theme.secondaryBackground }]}>
          <View style={styles.sectionHeader}>
            <Icon name="trophy" size={20} color={theme.primary} />
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Recent Achievements</Text>
          </View>
          
          <TouchableOpacity 
            style={styles.achievementCard}
            onPress={() => navigation.navigate('Achievements')}
          >
            <Icon 
              name={unlockedAchievements[unlockedAchievements.length - 1].icon || 'trophy'} 
              size={30} 
              color={theme.accent} 
            />
            <View style={styles.achievementInfo}>
              <Text style={[styles.achievementTitle, { color: theme.text }]}>
                {unlockedAchievements[unlockedAchievements.length - 1].title}
              </Text>
              <Text style={[styles.achievementDesc, { color: theme.text }]}>
                {unlockedAchievements[unlockedAchievements.length - 1].description}
              </Text>
            </View>
            <Icon name="chevron-right" size={24} color={theme.text} />
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  userInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatar: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  userStats: {
    marginLeft: 12,
  },
  username: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  progressContainer: {
    width: 150,
  },
  progressText: {
    fontSize: 12,
    marginTop: 2,
  },
  pointsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 8,
    borderRadius: 16,
  },
  pointsText: {
    color: '#fff',
    fontWeight: 'bold',
    marginLeft: 4,
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
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  focusButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 8,
    marginTop: 12,
  },
  focusButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    marginLeft: 8,
  },
  addButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 8,
    marginTop: 12,
  },
  addButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    marginLeft: 8,
  },
  emptyState: {
    alignItems: 'center',
    padding: 24,
  },
  emptyStateText: {
    marginTop: 8,
    textAlign: 'center',
  },
  achievementCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 8,
    backgroundColor: 'rgba(0,0,0,0.05)',
  },
  achievementInfo: {
    flex: 1,
    marginLeft: 12,
    marginRight: 12,
  },
  achievementTitle: {
    fontWeight: 'bold',
    fontSize: 16,
  },
  achievementDesc: {
    fontSize: 14,
    opacity: 0.8,
  },
});

export default HomeScreen;
