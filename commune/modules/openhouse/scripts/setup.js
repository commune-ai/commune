#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  cyan: '\x1b[36m',
  yellow: '\x1b[33m',
  red: '\x1b[31m'
};

// Print banner
console.log(`${colors.bright}${colors.cyan}======================================${colors.reset}`);
console.log(`${colors.bright}${colors.cyan}  Setting up ADHD Hero Application  ${colors.reset}`);
console.log(`${colors.bright}${colors.cyan}======================================${colors.reset}\n`);

// Function to run shell commands
function runCommand(command) {
  try {
    console.log(`${colors.yellow}Running: ${command}${colors.reset}`);
    execSync(command, { stdio: 'inherit' });
    return true;
  } catch (error) {
    console.error(`${colors.red}Failed to execute ${command}${colors.reset}`, error);
    return false;
  }
}

// Check if React Native CLI is installed
function checkReactNativeCLI() {
  try {
    execSync('npx react-native --version', { stdio: 'ignore' });
    console.log(`${colors.green}✓ React Native CLI is available${colors.reset}`);
    return true;
  } catch (error) {
    console.log(`${colors.yellow}! React Native CLI not found. Installing...${colors.reset}`);
    return runCommand('npm install -g react-native-cli');
  }
}

// Initialize a new React Native project
function initializeProject() {
  const projectName = 'ADHDHero';
  
  if (fs.existsSync(projectName)) {
    console.log(`${colors.yellow}! Directory ${projectName} already exists. Skipping initialization.${colors.reset}`);
    process.chdir(projectName);
    return true;
  }
  
  console.log(`${colors.cyan}Initializing new React Native project: ${projectName}${colors.reset}`);
  const success = runCommand(`npx react-native init ${projectName}`);
  
  if (success) {
    process.chdir(projectName);
    console.log(`${colors.green}✓ Project initialized successfully${colors.reset}`);
    return true;
  }
  
  return false;
}

// Install dependencies
function installDependencies() {
  console.log(`${colors.cyan}Installing dependencies...${colors.reset}`);
  
  const dependencies = [
    '@react-native-community/masked-view',
    '@react-navigation/bottom-tabs',
    '@react-navigation/native',
    '@react-navigation/stack',
    'firebase',
    'react-native-gesture-handler',
    'react-native-reanimated',
    'react-native-safe-area-context',
    'react-native-screens',
    'react-native-svg',
    'react-native-vector-icons',
    'react-redux',
    'redux',
    'redux-thunk'
  ];
  
  return runCommand(`npm install ${dependencies.join(' ')}`);
}

// Copy source files
function copySourceFiles() {
  console.log(`${colors.cyan}Setting up project structure...${colors.reset}`);
  
  // Create necessary directories
  const directories = [
    'src',
    'src/components',
    'src/screens',
    'src/redux',
    'src/redux/actions',
    'src/redux/reducers',
    'src/context',
    'src/navigation'
  ];
  
  directories.forEach(dir => {
    const dirPath = path.join(process.cwd(), dir);
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
  });
  
  console.log(`${colors.green}✓ Project structure created${colors.reset}`);
  return true;
}

// Main function
async function main() {
  try {
    // Check prerequisites
    if (!checkReactNativeCLI()) {
      throw new Error('Failed to install React Native CLI');
    }
    
    // Initialize project
    if (!initializeProject()) {
      throw new Error('Failed to initialize React Native project');
    }
    
    // Install dependencies
    if (!installDependencies()) {
      throw new Error('Failed to install dependencies');
    }
    
    // Copy source files
    if (!copySourceFiles()) {
      throw new Error('Failed to set up project structure');
    }
    
    // Success message
    console.log(`\n${colors.bright}${colors.green}======================================${colors.reset}`);
    console.log(`${colors.bright}${colors.green}  ADHD Hero setup completed successfully!  ${colors.reset}`);
    console.log(`${colors.bright}${colors.green}======================================${colors.reset}\n`);
    
    console.log(`${colors.cyan}To start the development server:${colors.reset}`);
    console.log(`  cd ADHDHero`);
    console.log(`  npx react-native start\n`);
    
    console.log(`${colors.cyan}To run on Android:${colors.reset}`);
    console.log(`  npx react-native run-android\n`);
    
    console.log(`${colors.cyan}To run on iOS:${colors.reset}`);
    console.log(`  npx react-native run-ios\n`);
    
  } catch (error) {
    console.error(`${colors.red}Error: ${error.message}${colors.reset}`);
    process.exit(1);
  }
}

// Run the main function
main();
