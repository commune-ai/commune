import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TextInput, 
  TouchableOpacity, 
  ScrollView,
  Switch,
} from 'react-native';
import { useDispatch } from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme } from '../context/ThemeContext';
import { addTask, updateTask } from '../redux/actions/taskActions';

const AddTaskScreen = ({ route, navigation }) => {
  const { theme } = useTheme();
  const dispatch = useDispatch();
  
  // Check if we're editing an existing task
  const existingTask = route.params?.task;
  const isEditing = !!existingTask;
  
  // Task state
  const [title, setTitle] = useState(existingTask?.title || '');
  const [description, setDescription] = useState(existingTask?.description || '');
  const [priority, setPriority] = useState(existingTask?.priority || 'medium');
  const [difficulty, setDifficulty] = useState(existingTask?.difficulty || 'medium');
  const [hasDueDate, setHasDueDate] = useState(!!existingTask?.dueDate);
  const [dueDate, setDueDate] = useState(
    existingTask?.dueDate ? new Date(existingTask.dueDate) : new Date()
  );
  const [hasEstimatedTime, setHasEstimatedTime] = useState(!!existingTask?.estimatedTime);
  const [estimatedTime, setEstimatedTime] = useState(existingTask?.estimatedTime?.toString() || '25');
  
  // Form validation
  const [titleError, setTitleError] = useState('');
  const [showDatePicker, setShowDatePicker] = useState(false);
  
  const validateForm = () => {
    let isValid = true;
    
    if (!title.trim()) {
      setTitleError('Title is required');
      isValid = false;
    } else {
      setTitleError('');
    }
    
    return isValid;
  };
  
  const handleSubmit = () => {
    if (!validateForm()) return;
    
    const taskData = {
      title,
      description,
      priority,
      difficulty,
      dueDate: hasDueDate ? dueDate.toISOString().split('T')[0] : null,
      estimatedTime: hasEstimatedTime ? parseInt(estimatedTime, 10) : null,
    };
    
    if (isEditing) {
      dispatch(updateTask({ ...existingTask, ...taskData }));
    } else {
      dispatch(addTask(taskData));
    }
    
    navigation.goBack();
  };
  
  // Priority selection buttons
  const PriorityButton = ({ value, label, color }) => (
    <TouchableOpacity 
      style={[styles.priorityButton, { 
        backgroundColor: priority === value ? color : theme.secondaryBackground,
        borderColor: color,
        borderWidth: 1,
      }]}
      onPress={() => setPriority(value)}
    >
      <Text style={[styles.priorityButtonText, { 
        color: priority === value ? '#fff' : color 
      }]}>{label}</Text>
    </TouchableOpacity>
  );
  
  // Difficulty selection buttons
  const DifficultyButton = ({ value, label }) => (
    <TouchableOpacity 
      style={[styles.difficultyButton, { 
        backgroundColor: difficulty === value ? theme.primary : theme.secondaryBackground,
        borderColor: theme.primary,
        borderWidth: 1,
      }]}
      onPress={() => setDifficulty(value)}
    >
      <Text style={[styles.difficultyButtonText, { 
        color: difficulty === value ? '#fff' : theme.primary 
      }]}>{label}</Text>
    </TouchableOpacity>
  );
  
  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.background }]}>
      <View style={styles.formContainer}>
        {/* Title Input */}
        <View style={styles.inputGroup}>
          <Text style={[styles.label, { color: theme.text }]}>Quest Title*</Text>
          <TextInput
            style={[styles.input, { 
              backgroundColor: theme.secondaryBackground, 
              color: theme.text,
              borderColor: titleError ? theme.error : theme.border,
            }]}
            placeholder="Enter quest title"
            placeholderTextColor="#888"
            value={title}
            onChangeText={setTitle}
          />
          {titleError ? <Text style={[styles.errorText, { color: theme.error }]}>{titleError}</Text> : null}
        </View>
        
        {/* Description Input */}
        <View style={styles.inputGroup}>
          <Text style={[styles.label, { color: theme.text }]}>Description</Text>
          <TextInput
            style={[styles.textArea, { backgroundColor: theme.secondaryBackground, color: theme.text }]}
            placeholder="Enter quest description"
            placeholderTextColor="#888"
            value={description}
            onChangeText={setDescription}
            multiline
            numberOfLines={4}
            textAlignVertical="top"
          />
        </View>
        
        {/* Priority Selection */}
        <View style={styles.inputGroup}>
          <Text style={[styles.label, { color: theme.text }]}>Priority</Text>
          <View style={styles.priorityContainer}>
            <PriorityButton value="low" label="Low" color={theme.success} />
            <PriorityButton value="medium" label="Medium" color={theme.warning} />
            <PriorityButton value="high" label="High" color={theme.error} />
          </View>
        </View>
        
        {/* Difficulty Selection */}
        <View style={styles.inputGroup}>
          <Text style={[styles.label, { color: theme.text }]}>Difficulty</Text>
          <View style={styles.difficultyContainer}>
            <DifficultyButton value="easy" label="Easy" />
            <DifficultyButton value="medium" label="Medium" />
            <DifficultyButton value="hard" label="Hard" />
          </View>
        </View>
        
        {/* Due Date */}
        <View style={styles.inputGroup}>
          <View style={styles.switchRow}>
            <Text style={[styles.label, { color: theme.text }]}>Due Date</Text>
            <Switch
              value={hasDueDate}
              onValueChange={setHasDueDate}
              trackColor={{ false: '#767577', true: theme.primary + '80' }}
              thumbColor={hasDueDate ? theme.primary : '#f4f3f4'}
            />
          </View>
          
          {hasDueDate && (
            <TouchableOpacity 
              style={[styles.dateButton, { backgroundColor: theme.secondaryBackground }]}
              onPress={() => setShowDatePicker(true)}
            >
              <Icon name="calendar" size={20} color={theme.primary} />
              <Text style={[styles.dateButtonText, { color: theme.text }]}>
                {dueDate.toLocaleDateString()}
              </Text>
            </TouchableOpacity>
          )}
          
          {/* Date picker would be implemented here */}
        </View>
        
        {/* Estimated Time */}
        <View style={styles.inputGroup}>
          <View style={styles.switchRow}>
            <Text style={[styles.label, { color: theme.text }]}>Estimated Time</Text>
            <Switch
              value={hasEstimatedTime}
              onValueChange={setHasEstimatedTime}
              trackColor={{ false: '#767577', true: theme.primary + '80' }}
              thumbColor={hasEstimatedTime ? theme.primary : '#f4f3f4'}
            />
          </View>
          
          {hasEstimatedTime && (
            <View style={styles.timeInputContainer}>
              <TextInput
                style={[styles.timeInput, { backgroundColor: theme.secondaryBackground, color: theme.text }]}
                placeholder="25"
                placeholderTextColor="#888"
                value={estimatedTime}
                onChangeText={setEstimatedTime}
                keyboardType="numeric"
              />
              <Text style={[styles.timeUnit, { color: theme.text }]}>minutes</Text>
            </View>
          )}
        </View>
        
        {/* Submit Button */}
        <TouchableOpacity 
          style={[styles.submitButton, { backgroundColor: theme.primary }]}
          onPress={handleSubmit}
        >
          <Text style={styles.submitButtonText}>{isEditing ? 'Update Quest' : 'Create Quest'}</Text>
        </TouchableOpacity>
        
        {/* Cancel Button */}
        <TouchableOpacity 
          style={[styles.cancelButton, { backgroundColor: 'transparent', borderColor: theme.border, borderWidth: 1 }]}
          onPress={() => navigation.goBack()}
        >
          <Text style={[styles.cancelButtonText, { color: theme.text }]}>Cancel</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  formContainer: {
    padding: 16,
  },
  inputGroup: {
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  input: {
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
  },
  textArea: {
    borderRadius: 8,
    padding: 12,
    minHeight: 100,
    borderWidth: 1,
  },
  errorText: {
    marginTop: 4,
    fontSize: 12,
  },
  priorityContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  priorityButton: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
    marginHorizontal: 4,
  },
  priorityButtonText: {
    fontWeight: 'bold',
  },
  difficultyContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  difficultyButton: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
    marginHorizontal: 4,
  },
  difficultyButtonText: {
    fontWeight: 'bold',
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  dateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 8,
    marginTop: 8,
  },
  dateButtonText: {
    marginLeft: 8,
  },
  timeInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  timeInput: {
    borderRadius: 8,
    padding: 12,
    width: 80,
    textAlign: 'center',
  },
  timeUnit: {
    marginLeft: 8,
    fontSize: 16,
  },
  submitButton: {
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 20,
  },
  submitButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  cancelButton: {
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 12,
  },
  cancelButtonText: {
    fontWeight: 'bold',
    fontSize: 16,
  },
});

export default AddTaskScreen;
