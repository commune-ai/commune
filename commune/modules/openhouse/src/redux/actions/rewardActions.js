// Action Types
export const ADD_REWARD = 'ADD_REWARD';
export const UPDATE_REWARD = 'UPDATE_REWARD';
export const DELETE_REWARD = 'DELETE_REWARD';
export const CLAIM_REWARD = 'CLAIM_REWARD';
export const SET_REWARDS = 'SET_REWARDS';

// Action Creators
export const addReward = (reward) => ({
  type: ADD_REWARD,
  payload: {
    ...reward,
    id: Date.now().toString(), // Simple ID generation
    createdAt: new Date().toISOString(),
  },
});

export const updateReward = (reward) => ({
  type: UPDATE_REWARD,
  payload: reward,
});

export const deleteReward = (rewardId) => ({
  type: DELETE_REWARD,
  payload: rewardId,
});

export const claimReward = (reward) => ({
  type: CLAIM_REWARD,
  payload: reward,
});

export const setRewards = (rewards) => ({
  type: SET_REWARDS,
  payload: rewards,
});

// Default rewards
export const initializeDefaultRewards = () => {
  return (dispatch) => {
    const defaultRewards = [
      {
        id: '1',
        title: '15 Minutes of Free Time',
        description: 'Take a guilt-free break for 15 minutes',
        cost: 50,
        icon: 'coffee',
        category: 'break',
        createdAt: new Date().toISOString(),
      },
      {
        id: '2',
        title: '30 Minutes of Gaming',
        description: 'Enjoy 30 minutes of your favorite game',
        cost: 100,
        icon: 'gamepad-variant',
        category: 'entertainment',
        createdAt: new Date().toISOString(),
      },
      {
        id: '3',
        title: 'Social Media Break',
        description: '15 minutes of social media browsing',
        cost: 75,
        icon: 'instagram',
        category: 'social',
        createdAt: new Date().toISOString(),
      },
      {
        id: '4',
        title: 'Treat Yourself',
        description: 'Buy yourself a small treat (under $5)',
        cost: 200,
        icon: 'food',
        category: 'self-care',
        createdAt: new Date().toISOString(),
      },
      {
        id: '5',
        title: 'Movie Night',
        description: 'Watch a movie of your choice',
        cost: 300,
        icon: 'movie',
        category: 'entertainment',
        createdAt: new Date().toISOString(),
      },
    ];
    
    dispatch(setRewards(defaultRewards));
  };
};
