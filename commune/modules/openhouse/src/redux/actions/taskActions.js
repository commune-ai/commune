// Action Types
export const ADD_TASK = 'ADD_TASK';
export const UPDATE_TASK = 'UPDATE_TASK';
export const DELETE_TASK = 'DELETE_TASK';
export const COMPLETE_TASK = 'COMPLETE_TASK';
export const SET_TASKS = 'SET_TASKS';
export const SET_TASK_PRIORITY = 'SET_TASK_PRIORITY';
export const START_TASK = 'START_TASK';
export const PAUSE_TASK = 'PAUSE_TASK';

// Action Creators
export const addTask = (task) => ({
  type: ADD_TASK,
  payload: {
    ...task,
    id: Date.now().toString(), // Simple ID generation
    createdAt: new Date().toISOString(),
    status: 'pending',
    completed: false,
    timeSpent: 0,
  },
});

export const updateTask = (task) => ({
  type: UPDATE_TASK,
  payload: task,
});

export const deleteTask = (taskId) => ({
  type: DELETE_TASK,
  payload: taskId,
});

export const completeTask = (taskId) => ({
  type: COMPLETE_TASK,
  payload: {
    id: taskId,
    completedAt: new Date().toISOString(),
  },
});

export const setTasks = (tasks) => ({
  type: SET_TASKS,
  payload: tasks,
});

export const setTaskPriority = (taskId, priority) => ({
  type: SET_TASK_PRIORITY,
  payload: {
    id: taskId,
    priority,
  },
});

export const startTask = (taskId) => ({
  type: START_TASK,
  payload: taskId,
});

export const pauseTask = (taskId) => ({
  type: PAUSE_TASK,
  payload: {
    id: taskId,
  },
});

// Thunk Actions for Async Operations
export const completeTaskAndUpdateStats = (taskId) => {
  return (dispatch, getState) => {
    const { tasks } = getState().tasks;
    const task = tasks.find(t => t.id === taskId);
    
    if (!task) return;
    
    // Calculate rewards based on task priority and difficulty
    const experiencePoints = calculateExperiencePoints(task);
    const points = calculatePoints(task);
    
    // Complete the task
    dispatch(completeTask(taskId));
    
    // Update user stats
    dispatch({ type: 'ADD_EXPERIENCE', payload: experiencePoints });
    dispatch({ type: 'ADD_POINTS', payload: points });
    
    // Check if user should level up
    const { stats } = getState().user;
    if (stats.experience + experiencePoints >= stats.experienceToNextLevel) {
      dispatch({ type: 'LEVEL_UP' });
    }
    
    // Check for achievements
    checkForAchievements(dispatch, getState);
    
    // Update streak if applicable
    updateDailyStreak(dispatch, getState);
  };
};

// Helper functions
const calculateExperiencePoints = (task) => {
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

const checkForAchievements = (dispatch, getState) => {
  const { tasks } = getState().tasks;
  const { stats } = getState().user;
  const { achievements, unlockedAchievements } = getState().achievements;
  
  const completedTasks = tasks.filter(task => task.completed).length;
  
  // Check for task completion achievements
  achievements.forEach(achievement => {
    if (unlockedAchievements.some(a => a.id === achievement.id)) return;
    
    let shouldUnlock = false;
    
    switch (achievement.type) {
      case 'tasks_completed':
        shouldUnlock = completedTasks >= achievement.requirement;
        break;
      case 'level_reached':
        shouldUnlock = stats.level >= achievement.requirement;
        break;
      case 'streak_days':
        shouldUnlock = stats.streak >= achievement.requirement;
        break;
      // Add more achievement types as needed
    }
    
    if (shouldUnlock) {
      dispatch({ 
        type: 'UNLOCK_ACHIEVEMENT', 
        payload: achievement 
      });
    }
  });
};

const updateDailyStreak = (dispatch, getState) => {
  const { stats } = getState().user;
  const today = new Date().toISOString().split('T')[0];
  
  // If last active was yesterday, increment streak
  if (stats.lastActive) {
    const lastActive = new Date(stats.lastActive).toISOString().split('T')[0];
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split('T')[0];
    
    if (lastActive === yesterdayStr || lastActive === today) {
      dispatch({
        type: 'UPDATE_STREAK',
        payload: {
          streak: stats.streak + 1,
          lastActive: today,
        },
      });
    } else if (lastActive !== today) {
      // Streak broken
      dispatch({
        type: 'UPDATE_STREAK',
        payload: {
          streak: 1, // Reset to 1 for today
          lastActive: today,
        },
      });
    }
  } else {
    // First time tracking streak
    dispatch({
      type: 'UPDATE_STREAK',
      payload: {
        streak: 1,
        lastActive: today,
      },
    });
  }
};
