import { 
  ADD_TASK, 
  UPDATE_TASK, 
  DELETE_TASK, 
  COMPLETE_TASK,
  SET_TASKS,
  SET_TASK_PRIORITY,
  START_TASK,
  PAUSE_TASK
} from '../actions/taskActions';

const initialState = {
  tasks: [],
  activeTask: null,
  loading: false,
  error: null,
};

const tasksReducer = (state = initialState, action) => {
  switch (action.type) {
    case SET_TASKS:
      return {
        ...state,
        tasks: action.payload,
        loading: false,
      };
    
    case ADD_TASK:
      return {
        ...state,
        tasks: [...state.tasks, action.payload],
      };
    
    case UPDATE_TASK:
      return {
        ...state,
        tasks: state.tasks.map(task => 
          task.id === action.payload.id ? { ...task, ...action.payload } : task
        ),
      };
    
    case DELETE_TASK:
      return {
        ...state,
        tasks: state.tasks.filter(task => task.id !== action.payload),
      };
    
    case COMPLETE_TASK:
      return {
        ...state,
        tasks: state.tasks.map(task => 
          task.id === action.payload.id 
            ? { ...task, completed: true, completedAt: action.payload.completedAt } 
            : task
        ),
        activeTask: state.activeTask && state.activeTask.id === action.payload.id ? null : state.activeTask,
      };
    
    case SET_TASK_PRIORITY:
      return {
        ...state,
        tasks: state.tasks.map(task => 
          task.id === action.payload.id 
            ? { ...task, priority: action.payload.priority } 
            : task
        ),
      };
    
    case START_TASK:
      return {
        ...state,
        activeTask: state.tasks.find(task => task.id === action.payload),
        tasks: state.tasks.map(task => 
          task.id === action.payload 
            ? { ...task, status: 'in_progress', startedAt: Date.now() } 
            : task
        ),
      };
    
    case PAUSE_TASK:
      return {
        ...state,
        activeTask: null,
        tasks: state.tasks.map(task => 
          task.id === action.payload.id 
            ? { 
                ...task, 
                status: 'paused', 
                timeSpent: (task.timeSpent || 0) + (Date.now() - (task.startedAt || Date.now())) / 1000 
              } 
            : task
        ),
      };
    
    default:
      return state;
  }
};

export default tasksReducer;
