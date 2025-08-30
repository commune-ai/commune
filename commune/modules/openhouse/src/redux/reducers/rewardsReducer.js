import { 
  ADD_REWARD, 
  UPDATE_REWARD, 
  DELETE_REWARD, 
  CLAIM_REWARD,
  SET_REWARDS
} from '../actions/rewardActions';

const initialState = {
  rewards: [],
  claimedRewards: [],
  loading: false,
  error: null,
};

const rewardsReducer = (state = initialState, action) => {
  switch (action.type) {
    case SET_REWARDS:
      return {
        ...state,
        rewards: action.payload,
        loading: false,
      };
    
    case ADD_REWARD:
      return {
        ...state,
        rewards: [...state.rewards, action.payload],
      };
    
    case UPDATE_REWARD:
      return {
        ...state,
        rewards: state.rewards.map(reward => 
          reward.id === action.payload.id ? { ...reward, ...action.payload } : reward
        ),
      };
    
    case DELETE_REWARD:
      return {
        ...state,
        rewards: state.rewards.filter(reward => reward.id !== action.payload),
      };
    
    case CLAIM_REWARD:
      return {
        ...state,
        claimedRewards: [...state.claimedRewards, {
          ...action.payload,
          claimedAt: new Date().toISOString(),
        }],
      };
    
    default:
      return state;
  }
};

export default rewardsReducer;
