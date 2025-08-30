import React from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity } from 'react-native';
import { useSelector } from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme } from '../context/ThemeContext';

const AchievementsScreen = () => {
  const { theme } = useTheme();
  const { achievements, unlockedAchievements } = useSelector(state => state.achievements);
  
  // Group achievements by type
  const groupedAchievements = achievements.reduce((acc, achievement) => {
    const type = achievement.type;
    if (!acc[type]) {
      acc[type] = [];
    }
    acc[type].push(achievement);
    return acc;
  }, {});
  
  // Check if achievement is unlocked
  const isUnlocked = (achievementId) => {
    return unlockedAchievements.some(a => a.id === achievementId);
  };
  
  // Format achievement type for display
  const formatType = (type) => {
    switch (type) {
      case 'tasks_completed':
        return 'Task Completion';
      case 'level_reached':
        return 'Level Milestones';
      case 'streak_days':
        return 'Consistency Streaks';
      default:
        return type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }
  };
  
  const renderAchievementItem = ({ item }) => {
    const unlocked = isUnlocked(item.id);
    
    return (
      <View style={[styles.achievementCard, { 
        backgroundColor: theme.secondaryBackground,
        opacity: unlocked ? 1 : 0.6,
      }]}>
        <View style={[styles.iconContainer, { 
          backgroundColor: unlocked ? theme.accent : theme.border 
        }]}>
          <Icon 
            name={unlocked ? item.icon : 'lock'} 
            size={24} 
            color="#fff" 
          />
        </View>
        
        <View style={styles.achievementInfo}>
          <Text style={[styles.achievementTitle, { color: theme.text }]}>
            {item.title}
          </Text>
          <Text style={[styles.achievementDesc, { color: theme.text }]}>
            {item.description}
          </Text>
          
          {item.reward && (
            <View style={styles.rewardContainer}>
              {item.reward.experience && (
                <View style={styles.rewardBadge}>
                  <Icon name="star" size={12} color={theme.accent} />
                  <Text style={[styles.rewardText, { color: theme.text }]}>
                    {item.reward.experience} XP
                  </Text>
                </View>
              )}
              
              {item.reward.points && (
                <View style={styles.rewardBadge}>
                  <Icon name="coin" size={12} color={theme.accent} />
                  <Text style={[styles.rewardText, { color: theme.text }]}>
                    {item.reward.points} Points
                  </Text>
                </View>
              )}
            </View>
          )}
        </View>
        
        {unlocked && (
          <View style={[styles.unlockedBadge, { backgroundColor: theme.success }]}>
            <Icon name="check" size={16} color="#fff" />
          </View>
        )}
      </View>
    );
  };
  
  const renderSectionHeader = ({ section }) => (
    <View style={styles.sectionHeader}>
      <Text style={[styles.sectionTitle, { color: theme.text }]}>
        {formatType(section.title)}
      </Text>
      <View style={[styles.progressBadge, { backgroundColor: theme.primary }]}>
        <Text style={styles.progressText}>
          {section.data.filter(item => isUnlocked(item.id)).length}/{section.data.length}
        </Text>
      </View>
    </View>
  );
  
  // Convert grouped achievements to sections format
  const sections = Object.keys(groupedAchievements).map(type => ({
    title: type,
    data: groupedAchievements[type],
  }));
  
  return (
    <View style={[styles.container, { backgroundColor: theme.background }]}>
      {/* Header Stats */}
      <View style={[styles.statsContainer, { backgroundColor: theme.primary }]}>
        <Text style={styles.statsTitle}>Achievements Unlocked</Text>
        <View style={styles.statsRow}>
          <Text style={styles.statsNumber}>{unlockedAchievements.length}</Text>
          <Text style={styles.statsTotal}>/ {achievements.length}</Text>
        </View>
        <View style={styles.progressBarBackground}>
          <View 
            style={[styles.progressBarFill, { 
              width: `${(unlockedAchievements.length / achievements.length) * 100}%` 
            }]}
          />
        </View>
      </View>
      
      {/* Sections */}
      {sections.map(section => (
        <View key={section.title}>
          {renderSectionHeader({ section })}
          <FlatList
            data={section.data}
            renderItem={renderAchievementItem}
            keyExtractor={item => item.id}
            scrollEnabled={false}
          />
        </View>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  statsContainer: {
    padding: 24,
    alignItems: 'center',
  },
  statsTitle: {
    color: '#fff',
    fontSize: 16,
    marginBottom: 8,
  },
  statsRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
  },
  statsNumber: {
    color: '#fff',
    fontSize: 36,
    fontWeight: 'bold',
  },
  statsTotal: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 18,
    marginLeft: 4,
  },
  progressBarBackground: {
    width: '80%',
    height: 8,
    backgroundColor: 'rgba(255,255,255,0.3)',
    borderRadius: 4,
    marginTop: 12,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: '#fff',
    borderRadius: 4,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  progressBadge: {
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 12,
  },
  progressText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 12,
  },
  achievementCard: {
    flexDirection: 'row',
    padding: 16,
    marginHorizontal: 16,
    marginBottom: 12,
    borderRadius: 12,
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  achievementInfo: {
    flex: 1,
  },
  achievementTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  achievementDesc: {
    fontSize: 14,
    opacity: 0.8,
    marginBottom: 8,
  },
  rewardContainer: {
    flexDirection: 'row',
  },
  rewardBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.05)',
    paddingVertical: 2,
    paddingHorizontal: 6,
    borderRadius: 10,
    marginRight: 8,
  },
  rewardText: {
    fontSize: 12,
    marginLeft: 4,
  },
  unlockedBadge: {
    position: 'absolute',
    top: 12,
    right: 12,
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default AchievementsScreen;
