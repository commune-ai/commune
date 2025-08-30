import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme } from '../context/ThemeContext';
import ProgressBar from '../components/ProgressBar';
import StatsCard from '../components/StatsCard';

const ProfileScreen = ({ navigation }) => {
  const { theme, themeType, toggleTheme } = useTheme();
  const dispatch = useDispatch();
  const { stats } = useSelector(state => state.user);
  const { tasks } = useSelector(state => state.tasks);
  const { unlockedAchievements } = useSelector(state => state.achievements);
  
  // Calculate stats
  const completedTasks = tasks.filter(task => task.completed).length;
  const totalTasks = tasks.length;
  const completionRate = totalTasks > 0 ? (completedTasks / totalTasks) * 100 : 0;
  
  const highPriorityCompleted = tasks.filter(task => task.completed && task.priority === 'high').length;
  const totalHighPriority = tasks.filter(task => task.priority === 'high').length;
  
  // Calculate level progress
  const levelProgress = (stats.experience / stats.experienceToNextLevel) * 100;
  
  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.background }]}>
      {/* Profile Header */}
      <View style={[styles.header, { backgroundColor: theme.primary }]}>
        <View style={styles.avatarContainer}>
          <View style={styles.avatar}>
            <Text style={styles.avatarText}>{stats.level}</Text>
          </View>
        </View>
        <Text style={styles.heroTitle}>Level {stats.level} Hero</Text>
        <View style={styles.levelContainer}>
          <ProgressBar 
            progress={levelProgress} 
            color="#fff" 
            backgroundColor="rgba(255,255,255,0.3)" 
          />
          <Text style={styles.levelText}>
            {stats.experience}/{stats.experienceToNextLevel} XP
          </Text>
        </View>
      </View>
      
      {/* Stats Section */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>Your Stats</Text>
        
        <View style={styles.statsGrid}>
          <StatsCard 
            icon="check-circle" 
            value={completedTasks} 
            label="Tasks Completed" 
            color={theme.success} 
          />
          <StatsCard 
            icon="fire" 
            value={stats.streak} 
            label="Day Streak" 
            color={theme.accent} 
          />
          <StatsCard 
            icon="trophy" 
            value={unlockedAchievements.length} 
            label="Achievements" 
            color={theme.primary} 
          />
          <StatsCard 
            icon="coin" 
            value={stats.points} 
            label="Points Earned" 
            color={theme.accent} 
          />
        </View>
        
        <View style={styles.progressSection}>
          <View style={styles.progressItem}>
            <View style={styles.progressHeader}>
              <Text style={[styles.progressLabel, { color: theme.text }]}>Task Completion Rate</Text>
              <Text style={[styles.progressValue, { color: theme.text }]}>{completionRate.toFixed(0)}%</Text>
            </View>
            <ProgressBar progress={completionRate} color={theme.success} />
          </View>
          
          <View style={styles.progressItem}>
            <View style={styles.progressHeader}>
              <Text style={[styles.progressLabel, { color: theme.text }]}>High Priority Completion</Text>
              <Text style={[styles.progressValue, { color: theme.text }]}>
                {totalHighPriority > 0 ? ((highPriorityCompleted / totalHighPriority) * 100).toFixed(0) : 0}%
              </Text>
            </View>
            <ProgressBar 
              progress={totalHighPriority > 0 ? (highPriorityCompleted / totalHighPriority) * 100 : 0} 
              color={theme.error} 
            />
          </View>
        </View>
      </View>
      
      {/* Achievements Section */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Achievements</Text>
          <TouchableOpacity onPress={() => navigation.navigate('Achievements')}>
            <Text style={[styles.seeAllText, { color: theme.primary }]}>See All</Text>
          </TouchableOpacity>
        </View>
        
        {unlockedAchievements.length > 0 ? (
          <View style={styles.achievementsContainer}>
            {unlockedAchievements.slice(0, 3).map(achievement => (
              <View 
                key={achievement.id} 
                style={[styles.achievementItem, { backgroundColor: theme.secondaryBackground }]}
              >
                <Icon name={achievement.icon || 'trophy'} size={24} color={theme.accent} />
                <Text style={[styles.achievementTitle, { color: theme.text }]}>
                  {achievement.title}
                </Text>
              </View>
            ))}
          </View>
        ) : (
          <View style={styles.emptyState}>
            <Icon name="trophy-outline" size={40} color={theme.primary} />
            <Text style={[styles.emptyStateText, { color: theme.text }]}>
              Complete tasks to unlock achievements!
            </Text>
          </View>
        )}
      </View>
      
      {/* Settings Section */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>Settings</Text>
        
        <View style={[styles.settingsCard, { backgroundColor: theme.secondaryBackground }]}>
          <TouchableOpacity style={styles.settingItem} onPress={() => navigation.navigate('Settings')}>
            <Icon name="cog" size={24} color={theme.primary} />
            <Text style={[styles.settingText, { color: theme.text }]}>App Settings</Text>
            <Icon name="chevron-right" size={24} color={theme.text} />
          </TouchableOpacity>
          
          <View style={[styles.divider, { backgroundColor: theme.border }]} />
          
          <View style={styles.settingItem}>
            <Icon name="palette" size={24} color={theme.primary} />
            <Text style={[styles.settingText, { color: theme.text }]}>Theme</Text>
            <View style={styles.themeButtons}>
              <TouchableOpacity 
                style={[styles.themeButton, themeType === 'light' && styles.activeTheme]}
                onPress={() => toggleTheme('light')}
              >
                <Text style={{ color: themeType === 'light' ? '#fff' : theme.text }}>Light</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={[styles.themeButton, themeType === 'dark' && styles.activeTheme]}
                onPress={() => toggleTheme('dark')}
              >
                <Text style={{ color: themeType === 'dark' ? '#fff' : theme.text }}>Dark</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={[styles.themeButton, themeType === 'adhd' && styles.activeTheme]}
                onPress={() => toggleTheme('adhd')}
              >
                <Text style={{ color: themeType === 'adhd' ? '#fff' : theme.text }}>ADHD</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    padding: 24,
    alignItems: 'center',
  },
  avatarContainer: {
    marginBottom: 12,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarText: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#6200EE',
  },
  heroTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  levelContainer: {
    width: '80%',
  },
  levelText: {
    color: '#fff',
    textAlign: 'center',
    marginTop: 4,
  },
  section: {
    padding: 16,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  seeAllText: {
    fontWeight: 'bold',
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  progressSection: {
    marginTop: 16,
  },
  progressItem: {
    marginBottom: 16,
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
  },
  achievementsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  achievementItem: {
    width: '31%',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  achievementTitle: {
    marginTop: 8,
    textAlign: 'center',
    fontSize: 12,
    fontWeight: 'bold',
  },
  emptyState: {
    alignItems: 'center',
    padding: 24,
  },
  emptyStateText: {
    marginTop: 8,
    textAlign: 'center',
  },
  settingsCard: {
    borderRadius: 12,
    overflow: 'hidden',
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  settingText: {
    flex: 1,
    marginLeft: 12,
    fontSize: 16,
  },
  divider: {
    height: 1,
    width: '100%',
  },
  themeButtons: {
    flexDirection: 'row',
  },
  themeButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 16,
    marginLeft: 8,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  activeTheme: {
    backgroundColor: '#6200EE',
    borderColor: '#6200EE',
  },
});

export default ProfileScreen;
