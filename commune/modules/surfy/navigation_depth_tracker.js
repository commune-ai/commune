/**
 * Navigation Depth Tracker
 * 
 * This module tracks the navigation depth (number of clicks from starting point)
 * and provides utilities to manage and visualize navigation paths.
 */

class NavigationDepthTracker {
  constructor() {
    this.startingPoint = null;
    this.currentDepth = 0;
    this.maxDepth = 0;
    this.navigationPath = [];
    this.visitedNodes = new Map(); // Maps node IDs to their depths
  }

  /**
   * Initialize the tracker with a starting point
   * @param {string} nodeId - ID of the starting node
   */
  initialize(nodeId) {
    this.startingPoint = nodeId;
    this.currentDepth = 0;
    this.maxDepth = 0;
    this.navigationPath = [{ id: nodeId, depth: 0 }];
    this.visitedNodes.clear();
    this.visitedNodes.set(nodeId, 0);
  }

  /**
   * Navigate to a new node, increasing the depth
   * @param {string} nodeId - ID of the node being navigated to
   * @param {boolean} isBackNavigation - Whether this is a back navigation
   * @returns {number} The current depth after navigation
   */
  navigateTo(nodeId, isBackNavigation = false) {
    if (!this.startingPoint) {
      this.initialize(nodeId);
      return 0;
    }

    if (isBackNavigation) {
      // Find the previous occurrence of this node in the path
      const previousIndex = this.navigationPath
        .slice(0, -1)
        .findIndex(node => node.id === nodeId);
      
      if (previousIndex >= 0) {
        // Trim the path back to this point
        this.navigationPath = this.navigationPath.slice(0, previousIndex + 1);
        this.currentDepth = this.navigationPath[previousIndex].depth;
      } else {
        // If not found, treat as a new navigation
        this.currentDepth++;
        this.navigationPath.push({ id: nodeId, depth: this.currentDepth });
      }
    } else {
      // Forward navigation
      this.currentDepth++;
      this.navigationPath.push({ id: nodeId, depth: this.currentDepth });
      
      // Update visited nodes map
      if (!this.visitedNodes.has(nodeId) || this.visitedNodes.get(nodeId) > this.currentDepth) {
        this.visitedNodes.set(nodeId, this.currentDepth);
      }
    }

    // Update max depth if needed
    if (this.currentDepth > this.maxDepth) {
      this.maxDepth = this.currentDepth;
    }

    return this.currentDepth;
  }

  /**
   * Go back one step in the navigation path
   * @returns {Object|null} The previous node or null if at the start
   */
  goBack() {
    if (this.navigationPath.length <= 1) {
      return null; // Already at the starting point
    }
    
    // Remove the current node
    this.navigationPath.pop();
    
    // Get the previous node
    const previousNode = this.navigationPath[this.navigationPath.length - 1];
    this.currentDepth = previousNode.depth;
    
    return previousNode;
  }

  /**
   * Get the current navigation depth
   * @returns {number} Current depth
   */
  getDepth() {
    return this.currentDepth;
  }

  /**
   * Get the maximum depth reached in this session
   * @returns {number} Maximum depth
   */
  getMaxDepth() {
    return this.maxDepth;
  }

  /**
   * Get the full navigation path
   * @returns {Array} Array of navigation steps with node IDs and depths
   */
  getPath() {
    return [...this.navigationPath];
  }

  /**
   * Get a map of all visited nodes with their minimum depths
   * @returns {Map} Map of node IDs to their depths
   */
  getVisitedNodes() {
    return new Map(this.visitedNodes);
  }

  /**
   * Generate a depth visualization for the current path
   * @returns {string} A string representation of the path with depth indicators
   */
  visualizePath() {
    return this.navigationPath.map(node => {
      const indent = '  '.repeat(node.depth);
      return `${indent}→ ${node.id} (depth: ${node.depth})`;
    }).join('\n');
  }
}

// Export the tracker
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { NavigationDepthTracker };
} else {
  // Browser environment
  window.NavigationDepthTracker = NavigationDepthTracker;
}