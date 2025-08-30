# Navigation Depth Tracker

This project implements a system for tracking navigation depths in user interfaces, where the depth represents the number of clicks or steps from a starting point.

## Overview

The Navigation Depth Tracker provides a way to:

- Track how deep a user has navigated from a starting point
- Record the full navigation path
- Support both forward and backward navigation
- Visualize the navigation path with depth indicators
- Track the minimum depth at which each node was visited

## Files

- `navigation_depth_tracker.js`: Core implementation of the depth tracking logic
- `navigation_depth_demo.html`: Interactive demo showing how depth tracking works

## How Depth is Calculated

Depth is calculated as follows:

1. The starting point has a depth of 0
2. Each forward navigation increases the depth by 1
3. Back navigation returns to the previous depth of that node
4. The system tracks both current depth and maximum depth reached

## Usage

```javascript
// Create a new tracker instance
const tracker = new NavigationDepthTracker();

// Initialize with a starting node
tracker.initialize('home');

// Navigate to new nodes (returns the current depth)
const depth = tracker.navigateTo('products'); // depth = 1
tracker.navigateTo('product-details'); // depth = 2
tracker.navigateTo('reviews'); // depth = 3

// Go back to previous node
const prevNode = tracker.goBack(); // Returns the 'product-details' node

// Get current depth
const currentDepth = tracker.getDepth(); // 2

// Get maximum depth reached
const maxDepth = tracker.getMaxDepth(); // 3

// Get full navigation path
const path = tracker.getPath();
// [{id: 'home', depth: 0}, {id: 'products', depth: 1}, {id: 'product-details', depth: 2}]

// Get all visited nodes with their minimum depths
const visitedNodes = tracker.getVisitedNodes();
// Map {'home' => 0, 'products' => 1, 'product-details' => 2, 'reviews' => 3}

// Visualize the path
const visualization = tracker.visualizePath();
/*
→ home (depth: 0)
  → products (depth: 1)
    → product-details (depth: 2)
*/
```

## Demo

Open `navigation_depth_demo.html` in a web browser to see an interactive demonstration of the navigation depth tracking. The demo shows:

- How depth changes as you navigate between nodes
- The visualization of the navigation path
- How back navigation affects the depth
- Tracking of maximum depth reached

## Implementation Details

The tracker maintains several key pieces of state:

- `startingPoint`: The ID of the initial node
- `currentDepth`: The current navigation depth
- `maxDepth`: The maximum depth reached in the session
- `navigationPath`: An array of all navigation steps with their depths
- `visitedNodes`: A map of all visited nodes with their minimum depths

The implementation handles both forward and backward navigation, with special logic to determine if a navigation is returning to a previously visited node.