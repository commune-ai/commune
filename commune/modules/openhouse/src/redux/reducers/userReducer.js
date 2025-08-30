import { 
  LOGIN_SUCCESS, 
  LOGOUT, 
  UPDATE_PROFILE,
  ADD_EXPERIENCE,
  LEVEL_UP,
  ADD_POINTS,
  SPEND_POINTS,
  UPDATE_STREAK,
  SET_USER_DATA
} from '../actions/userActions';

const initialState = {
  user: null,
  isAuthenticated: false,
  loading: false,
  error: null,
  stats: {
    level: 1,
    experience: 0,
    experienceToNextLevel: 100,
    points: 0,
    streak: 0,
    lastActive: null,
    tasksCompleted: 0,
    focusTime: 0, // in minutes
  },
};

const userReducer = (state = initialState, action) => {
  switch (action.type) {
    case LOGIN_SUCCESS:
      return {
        ...state,
        user: action.payload,
        isAuthenticated: true,
        loading: false,
      };
    
    case LOGOUT:
      return {
        ...initialState,
      };
    
    case UPDATE_PROFILE:
      return {
        ...state,
        user: { ...state.user, ...action.payload },
      };
    
    case SET_USER_DATA:
      return {
        ...state,
        user: action.payload.user,
        isAuthenticated: true,
        stats: action.payload.stats || state.stats,
        loading: false,
      };
    
    case ADD_EXPERIENCE:
      const newExperience = state.stats.experience + action.payload;
      const needsLevelUp = newExperience >= state.stats.experienceToNextLevel;
      
      return {
        ...state,
        stats: {
          ...state.stats,
          experience: needsLevelUp ? newExperience - state.stats.experienceToNextLevel : newExperience,
        },
      };
    
    case LEVEL_UP:
      return {
        ...state,
        stats: {
          ...state.stats,
          level: state.stats.level + 1,
          experienceToNextLevel: Math.floor(state.stats.experienceToNextLevel * 1.5),
        },
      };
    
    case ADD_POINTS:
      return {
        ...state,
        stats: {
          ...state.stats,
          points: state.stats.points + action.payload,
        },
      };
    
    case SPEND_POINTS:
      return {
        ...state,
        stats: {
          ...state.stats,
          points: Math.max(0, state.stats.points - action.payload),
        },
      };
    
    case UPDATE_STREAK:
      return {
        ...state,
        stats: {
          ...state.stats,
          streak: action.payload.streak,
          lastActive: action.payload.lastActive,
        },
      };
    
    default:
      return state;
  }
};

export default userReducer;
