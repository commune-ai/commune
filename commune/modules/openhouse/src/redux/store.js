import { createStore, applyMiddleware, combineReducers } from 'redux';
import thunk from 'redux-thunk';
import tasksReducer from './reducers/tasksReducer';
import userReducer from './reducers/userReducer';
import rewardsReducer from './reducers/rewardsReducer';
import achievementsReducer from './reducers/achievementsReducer';

const rootReducer = combineReducers({
  tasks: tasksReducer,
  user: userReducer,
  rewards: rewardsReducer,
  achievements: achievementsReducer,
});

export const store = createStore(rootReducer, applyMiddleware(thunk));
