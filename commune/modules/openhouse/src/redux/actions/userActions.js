// Action Types
export const LOGIN_SUCCESS = 'LOGIN_SUCCESS';
export const LOGOUT = 'LOGOUT';
export const UPDATE_PROFILE = 'UPDATE_PROFILE';
export const ADD_EXPERIENCE = 'ADD_EXPERIENCE';
export const LEVEL_UP = 'LEVEL_UP';
export const ADD_POINTS = 'ADD_POINTS';
export const SPEND_POINTS = 'SPEND_POINTS';
export const UPDATE_STREAK = 'UPDATE_STREAK';
export const SET_USER_DATA = 'SET_USER_DATA';

// Action Creators
export const loginSuccess = (user) => ({
  type: LOGIN_SUCCESS,
  payload: user,
});

export const logout = () => ({
  type: LOGOUT,
});

export const updateProfile = (userData) => ({
  type: UPDATE_PROFILE,
  payload: userData,
});

export const addExperience = (amount) => ({
  type: ADD_EXPERIENCE,
  payload: amount,
});

export const levelUp = () => ({
  type: LEVEL_UP,
});

export const addPoints = (amount) => ({
  type: ADD_POINTS,
  payload: amount,
});

export const spendPoints = (amount) => ({
  type: SPEND_POINTS,
  payload: amount,
});

export const updateStreak = (streak, lastActive) => ({
  type: UPDATE_STREAK,
  payload: {
    streak,
    lastActive,
  },
});

export const setUserData = (userData) => ({
  type: SET_USER_DATA,
  payload: userData,
});

// Thunk Actions
export const claimReward = (reward) => {
  return (dispatch, getState) => {
    const { stats } = getState().user;
    
    // Check if user has enough points
    if (stats.points >= reward.cost) {
      // Deduct points
      dispatch(spendPoints(reward.cost));
      
      // Add the reward to claimed rewards
      dispatch({
        type: 'CLAIM_REWARD',
        payload: reward,
      });
      
      return true;
    }
    
    return false;
  };
};

export const checkDailyStreak = () => {
  return (dispatch, getState) => {
    const { stats } = getState().user;
    const today = new Date().toISOString().split('T')[0];
    
    if (!stats.lastActive) {
      // First login, set streak to 1
      dispatch(updateStreak(1, today));
      return;
    }
    
    const lastActive = new Date(stats.lastActive).toISOString().split('T')[0];
    
    if (lastActive === today) {
      // Already logged in today, do nothing
      return;
    }
    
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split('T')[0];
    
    if (lastActive === yesterdayStr) {
      // Logged in yesterday, increment streak
      dispatch(updateStreak(stats.streak + 1, today));
      
      // Bonus points for maintaining streak
      const streakBonus = Math.min(50, stats.streak * 5); // Cap at 50 points
      dispatch(addPoints(streakBonus));
    } else {
      // Streak broken, reset to 1
      dispatch(updateStreak(1, today));
    }
  };
};
