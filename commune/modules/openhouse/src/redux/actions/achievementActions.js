// Action Types
export const UNLOCK_ACHIEVEMENT = 'UNLOCK_ACHIEVEMENT';
export const SET_ACHIEVEMENTS = 'SET_ACHIEVEMENTS';

// Action Creators
export const unlockAchievement = (achievement) => ({
  type: UNLOCK_ACHIEVEMENT,
  payload: achievement,
});

export const setAchievements = (achievements) => ({
  type: SET_ACHIEVEMENTS,
  payload: achievements,
});

// Default achievements
export const initializeDefaultAchievements = () => {
  return (dispatch) => {
    const defaultAchievements = [
      {
        id: '1',
        title: 'First Steps',
        description: 'Complete your first task',
        icon: 'shoe-print',
        type: 'tasks_completed',
        requirement: 1,
        reward: {
          experience: 50,
          points: 25,
        },
      },
      {
        id: '2',
        title: 'Task Master',
        description: 'Complete 10 tasks',
        icon: 'check-all',
        type: 'tasks_completed',
        requirement: 10,
        reward: {
          experience: 100,
          points: 50,
        },
      },
      {
        id: '3',
        title: 'Productivity Ninja',
        description: 'Complete 50 tasks',
        icon: 'ninja',
        type: 'tasks_completed',
        requirement: 50,
        reward: {
          experience: 250,
          points: 125,
        },
      },
      {
        id: '4',
        title: 'Level 5 Hero',
        description: 'Reach level 5',
        icon: 'star',
        type: 'level_reached',
        requirement: 5,
        reward: {
          points: 100,
        },
      },
      {
        id: '5',
        title: 'Level 10 Champion',
        description: 'Reach level 10',
        icon: 'crown',
        type: 'level_reached',
        requirement: 10,
        reward: {
          points: 200,
        },
      },
      {
        id: '6',
        title: 'Consistency is Key',
        description: 'Maintain a 3-day streak',
        icon: 'calendar-check',
        type: 'streak_days',
        requirement: 3,
        reward: {
          experience: 75,
          points: 50,
        },
      },
      {
        id: '7',
        title: 'Week Warrior',
        description: 'Maintain a 7-day streak',
        icon: 'calendar-week',
        type: 'streak_days',
        requirement: 7,
        reward: {
          experience: 150,
          points: 100,
        },
      },
      {
        id: '8',
        title: 'Month Master',
        description: 'Maintain a 30-day streak',
        icon: 'calendar-month',
        type: 'streak_days',
        requirement: 30,
        reward: {
          experience: 500,
          points: 300,
        },
      },
    ];
    
    dispatch(setAchievements(defaultAchievements));
  };
};

// Apply achievement rewards
export const applyAchievementRewards = (achievementId) => {
  return (dispatch, getState) => {
    const { achievements } = getState().achievements;
    const achievement = achievements.find(a => a.id === achievementId);
    
    if (!achievement || !achievement.reward) return;
    
    if (achievement.reward.experience) {
      dispatch({ type: 'ADD_EXPERIENCE', payload: achievement.reward.experience });
    }
    
    if (achievement.reward.points) {
      dispatch({ type: 'ADD_POINTS', payload: achievement.reward.points });
    }
    
    // Check if user should level up
    const { stats } = getState().user;
    if (stats.experience >= stats.experienceToNextLevel) {
      dispatch({ type: 'LEVEL_UP' });
    }
  };
};
