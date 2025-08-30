import React, { useState } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity } from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme } from '../context/ThemeContext';
import TaskCard from '../components/TaskCard';
import FilterChip from '../components/FilterChip';

const QuestsScreen = ({ navigation }) => {
  const { theme } = useTheme();
  const dispatch = useDispatch();
  const { tasks } = useSelector(state => state.tasks);
  
  const [filter, setFilter] = useState('all'); // 'all', 'active', 'completed'
  const [priorityFilter, setPriorityFilter] = useState('all'); // 'all', 'high', 'medium', 'low'
  
  // Apply filters
  const filteredTasks = tasks.filter(task => {
    // Status filter
    if (filter === 'active' && task.completed) return false;
    if (filter === 'completed' && !task.completed) return false;
    
    // Priority filter
    if (priorityFilter !== 'all' && task.priority !== priorityFilter) return false;
    
    return true;
  });
  
  // Sort tasks: high priority first, then by due date
  const sortedTasks = [...filteredTasks].sort((a, b) => {
    // First sort by completion status
    if (a.completed && !b.completed) return 1;
    if (!a.completed && b.completed) return -1;
    
    // Then sort by priority for incomplete tasks
    if (!a.completed && !b.completed) {
      const priorityOrder = { high: 0, medium: 1, low: 2 };
      if (priorityOrder[a.priority] !== priorityOrder[b.priority]) {
        return priorityOrder[a.priority] - priorityOrder[b.priority];
      }
      
      // Finally sort by due date if priority is the same
      if (a.dueDate && b.dueDate) {
        return new Date(a.dueDate) - new Date(b.dueDate);
      }
    }
    
    // For completed tasks, sort by completion date (most recent first)
    if (a.completed && b.completed) {
      return new Date(b.completedAt) - new Date(a.completedAt);
    }
    
    return 0;
  });
  
  return (
    <View style={[styles.container, { backgroundColor: theme.background }]}>
      {/* Filter Section */}
      <View style={styles.filterContainer}>
        <Text style={[styles.filterLabel, { color: theme.text }]}>Status:</Text>
        <View style={styles.chipContainer}>
          <FilterChip 
            label="All" 
            selected={filter === 'all'} 
            onPress={() => setFilter('all')} 
          />
          <FilterChip 
            label="Active" 
            selected={filter === 'active'} 
            onPress={() => setFilter('active')} 
          />
          <FilterChip 
            label="Completed" 
            selected={filter === 'completed'} 
            onPress={() => setFilter('completed')} 
          />
        </View>
      </View>
      
      <View style={styles.filterContainer}>
        <Text style={[styles.filterLabel, { color: theme.text }]}>Priority:</Text>
        <View style={styles.chipContainer}>
          <FilterChip 
            label="All" 
            selected={priorityFilter === 'all'} 
            onPress={() => setPriorityFilter('all')} 
          />
          <FilterChip 
            label="High" 
            selected={priorityFilter === 'high'} 
            onPress={() => setPriorityFilter('high')} 
            color={theme.error}
          />
          <FilterChip 
            label="Medium" 
            selected={priorityFilter === 'medium'} 
            onPress={() => setPriorityFilter('medium')} 
            color={theme.warning}
          />
          <FilterChip 
            label="Low" 
            selected={priorityFilter === 'low'} 
            onPress={() => setPriorityFilter('low')} 
            color={theme.success}
          />
        </View>
      </View>
      
      {/* Tasks List */}
      {sortedTasks.length > 0 ? (
        <FlatList
          data={sortedTasks}
          keyExtractor={item => item.id}
          renderItem={({ item }) => (
            <TaskCard 
              task={item} 
              completed={item.completed}
              onPress={() => navigation.navigate('TaskDetail', { taskId: item.id })} 
            />
          )}
          contentContainerStyle={styles.listContent}
        />
      ) : (
        <View style={styles.emptyState}>
          <Icon name="sword-cross" size={60} color={theme.primary} />
          <Text style={[styles.emptyStateTitle, { color: theme.text }]}>No Quests Found</Text>
          <Text style={[styles.emptyStateText, { color: theme.text }]}>
            {filter === 'completed' 
              ? "You haven't completed any quests yet."
              : "You don't have any quests matching these filters."}
          </Text>
        </View>
      )}
      
      {/* Add Task Button */}
      <TouchableOpacity 
        style={[styles.addButton, { backgroundColor: theme.primary }]}
        onPress={() => navigation.navigate('AddTask')}
      >
        <Icon name="plus" size={24} color="#fff" />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  filterContainer: {
    marginBottom: 12,
  },
  filterLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  listContent: {
    paddingBottom: 80, // Space for FAB
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  emptyStateTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 16,
  },
  emptyStateText: {
    fontSize: 16,
    textAlign: 'center',
    marginTop: 8,
    opacity: 0.7,
  },
  addButton: {
    position: 'absolute',
    bottom: 24,
    right: 24,
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 3,
  },
});

export default QuestsScreen;
