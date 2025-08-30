import { 
  UNLOCK_ACHIEVEMENT, 
  SET_ACHIEVEMENTS 
} from '../actions/achievementActions';

const initialState = {
  achievements: [],
  unlockedAchievements: [],
  loading: false,
  error: null,
};

const achievementsReducer = (state = initialState, action) => {
  switch (action.type) {
    case SET_ACHIEVEMENTS:
      return {
        ...state,
        achievements: action.payload,
        loading: false,
      };
    
    case UNLOCK_ACHIEVEMENT:
      // Check if achievement is already unlocked
      if (state.unlockedAchievements.some(a => a.id === action.payload.id)) {
        return state;
      }
      
      return {
        ...state,
        unlockedAchievements: [...state.unlockedAchievements, {
          ...action.payload,
          unlockedAt: new Date().toISOString(),
        }],
      };
    
    default:
      return state;
  }
};

export default achievementsReducer;
