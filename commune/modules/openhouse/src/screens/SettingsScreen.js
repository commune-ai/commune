import React from 'react';
import { View, Text, StyleSheet, Switch, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { useDispatch } from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme } from '../context/ThemeContext';

const SettingsScreen = () => {
  const { theme } = useTheme();
  const dispatch = useDispatch();
  
  // Settings state
  const [notifications, setNotifications] = React.useState(true);
  const [soundEffects, setSoundEffects] = React.useState(true);
  const [vibration, setVibration] = React.useState(true);
  const [focusReminders, setFocusReminders] = React.useState(true);
  const [dailyGoalReminders, setDailyGoalReminders] = React.useState(true);
  const [autoStartBreaks, setAutoStartBreaks] = React.useState(false);
  const [focusDuration, setFocusDuration] = React.useState(25); // minutes
  const [shortBreakDuration, setShortBreakDuration] = React.useState(5); // minutes
  const [longBreakDuration, setLongBreakDuration] = React.useState(15); // minutes
  
  const resetProgress = () => {
    Alert.alert(
      'Reset Progress',
      'Are you sure you want to reset all your progress? This will reset your level, points, and achievements but keep your tasks.',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => {
            // Reset user stats
            dispatch({
              type: 'SET_USER_DATA',
              payload: {
                user: null,
                stats: {
                  level: 1,
                  experience: 0,
                  experienceToNextLevel: 100,
                  points: 0,
                  streak: 0,
                  lastActive: null,
                  tasksCompleted: 0,
                  focusTime: 0,
                },
              },
            });
            
            // Reset achievements
            dispatch({
              type: 'SET_ACHIEVEMENTS',
              payload: [],
            });
            
            Alert.alert('Success', 'Your progress has been reset.');
          },
        },
      ],
    );
  };
  
  const SettingSwitch = ({ value, onValueChange, label, description }) => (
    <View style={styles.settingRow}>
      <View style={styles.settingInfo}>
        <Text style={[styles.settingLabel, { color: theme.text }]}>{label}</Text>
        {description && <Text style={[styles.settingDescription, { color: theme.text }]}>{description}</Text>}
      </View>
      <Switch
        value={value}
        onValueChange={onValueChange}
        trackColor={{ false: '#767577', true: theme.primary + '80' }}
        thumbColor={value ? theme.primary : '#f4f3f4'}
      />
    </View>
  );
  
  const SettingButton = ({ onPress, label, icon, color, destructive }) => (
    <TouchableOpacity 
      style={[styles.settingButton, { borderColor: theme.border }]}
      onPress={onPress}
    >
      <View style={styles.settingButtonContent}>
        <Icon name={icon} size={24} color={destructive ? theme.error : color || theme.primary} />
        <Text style={[styles.settingButtonLabel, { 
          color: destructive ? theme.error : theme.text 
        }]}>{label}</Text>
      </View>
      <Icon name="chevron-right" size={24} color={theme.text} />
    </TouchableOpacity>
  );
  
  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.background }]}>
      {/* Notifications Section */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>Notifications</Text>
        
        <View style={[styles.settingCard, { backgroundColor: theme.secondaryBackground }]}>
          <SettingSwitch
            label="Notifications"
            description="Enable all app notifications"
            value={notifications}
            onValueChange={setNotifications}
          />
          
          <View style={[styles.divider, { backgroundColor: theme.border }]} />
          
          <SettingSwitch
            label="Focus Reminders"
            description="Remind you to focus on your tasks"
            value={focusReminders}
            onValueChange={setFocusReminders}
          />
          
          <View style={[styles.divider, { backgroundColor: theme.border }]} />
          
          <SettingSwitch
            label="Daily Goal Reminders"
            description="Remind you to complete daily goals"
            value={dailyGoalReminders}
            onValueChange={setDailyGoalReminders}
          />
        </View>
      </View>
      
      {/* Focus Timer Settings */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>Focus Timer</Text>
        
        <View style={[styles.settingCard, { backgroundColor: theme.secondaryBackground }]}>
          <SettingSwitch
            label="Sound Effects"
            description="Play sounds for timer events"
            value={soundEffects}
            onValueChange={setSoundEffects}
          />
          
          <View style={[styles.divider, { backgroundColor: theme.border }]} />
          
          <SettingSwitch
            label="Vibration"
            description="Vibrate on timer completion"
            value={vibration}
            onValueChange={setVibration}
          />
          
          <View style={[styles.divider, { backgroundColor: theme.border }]} />
          
          <SettingSwitch
            label="Auto-start Breaks"
            description="Automatically start breaks after focus sessions"
            value={autoStartBreaks}
            onValueChange={setAutoStartBreaks}
          />
        </View>
      </View>
      
      {/* Data Management */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>Data Management</Text>
        
        <View style={[styles.settingCard, { backgroundColor: theme.secondaryBackground }]}>
          <SettingButton
            label="Export Data"
            icon="export"
            onPress={() => Alert.alert('Coming Soon', 'This feature will be available in a future update.')}
          />
          
          <View style={[styles.divider, { backgroundColor: theme.border }]} />
          
          <SettingButton
            label="Import Data"
            icon="import"
            onPress={() => Alert.alert('Coming Soon', 'This feature will be available in a future update.')}
          />
          
          <View style={[styles.divider, { backgroundColor: theme.border }]} />
          
          <SettingButton
            label="Reset Progress"
            icon="refresh"
            destructive
            onPress={resetProgress}
          />
        </View>
      </View>
      
      {/* About */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>About</Text>
        
        <View style={[styles.settingCard, { backgroundColor: theme.secondaryBackground }]}>
          <SettingButton
            label="Help & Support"
            icon="help-circle"
            onPress={() => Alert.alert('Coming Soon', 'This feature will be available in a future update.')}
          />
          
          <View style={[styles.divider, { backgroundColor: theme.border }]} />
          
          <SettingButton
            label="Privacy Policy"
            icon="shield"
            onPress={() => Alert.alert('Coming Soon', 'This feature will be available in a future update.')}
          />
          
          <View style={[styles.divider, { backgroundColor: theme.border }]} />
          
          <SettingButton
            label="Terms of Service"
            icon="file-document"
            onPress={() => Alert.alert('Coming Soon', 'This feature will be available in a future update.')}
          />
          
          <View style={[styles.divider, { backgroundColor: theme.border }]} />
          
          <View style={styles.versionContainer}>
            <Text style={[styles.versionText, { color: theme.text }]}>Version 1.0.0</Text>
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
  section: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  settingCard: {
    borderRadius: 12,
    overflow: 'hidden',
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  settingInfo: {
    flex: 1,
    marginRight: 16,
  },
  settingLabel: {
    fontSize: 16,
    marginBottom: 4,
  },
  settingDescription: {
    fontSize: 14,
    opacity: 0.7,
  },
  divider: {
    height: 1,
    width: '100%',
  },
  settingButton: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 0,
  },
  settingButtonContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  settingButtonLabel: {
    fontSize: 16,
    marginLeft: 12,
  },
  versionContainer: {
    padding: 16,
    alignItems: 'center',
  },
  versionText: {
    fontSize: 14,
    opacity: 0.7,
  },
});

export default SettingsScreen;
