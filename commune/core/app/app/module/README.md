# Module Explorer Components

## Overview

This directory contains components for displaying code files with an interactive split-panel interface.

## Components

### ModuleExplorer

A split-panel file explorer that displays:
- **Left Panel**: List of files with their names and SHA-256 hash (first 8 characters)
- **Right Panel**: Selected file's code with syntax highlighting
- **Adjustable Divider**: Drag to resize panels (15% to 85% range)

#### Usage

```tsx
import { ModuleExplorer } from './ModuleExplorer'

const files = {
  '/path/to/file.tsx': 'file content here',
  '/path/to/another.ts': 'another file content'
}

<ModuleExplorer files={files} />
```

### ModuleContent

A collapsible code display component with:
- Syntax highlighting based on file extension
- Copy button functionality
- Line count display
- Expandable/collapsible view
- Preview mode when collapsed

#### Props

- `code`: The code content to display
- `path`: File path (used for display)
- `language`: Programming language for syntax highlighting
- `defaultExpanded`: Whether to expand by default (optional)

## Features

1. **File Hash Display**: Each file shows a truncated SHA-256 hash for quick identification
2. **Responsive Divider**: Click and drag the divider to adjust panel sizes
3. **Syntax Highlighting**: Automatic language detection based on file extension
4. **Selected State**: Visual feedback for the currently selected file
5. **Overflow Handling**: Proper scrolling for long file lists and code content

## Styling

- Uses Tailwind CSS for styling
- Green-themed UI with black background
- Smooth transitions and hover effects
- Responsive design