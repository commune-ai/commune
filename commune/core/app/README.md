# Simple Web App with Pagination

A clean, minimalist web application featuring a simple header and pagination system.

## Features

- **Simple Header**: Clean navigation bar with sticky positioning
- **Pagination**: Easy-to-use pagination with Previous/Next buttons
- **Responsive Design**: Works well on desktop and mobile devices
- **Keyboard Navigation**: Use arrow keys to navigate between pages
- **Clean UI**: Minimalist design with subtle shadows and smooth transitions

## Structure

- `index.html` - Main HTML structure
- `styles.css` - All styling for the application
- `script.js` - JavaScript for pagination functionality

## How It Works

1. The app displays 5 items per page
2. Navigation is handled through Previous/Next buttons
3. Current page information is displayed between the buttons
4. Buttons are automatically disabled when at the first or last page
5. Keyboard shortcuts (arrow keys) are supported for navigation

## Usage

Simply open `index.html` in a web browser. No build process or dependencies required.

## Customization

- Change `ITEMS_PER_PAGE` in `script.js` to adjust items shown per page
- Modify the `generateSampleData()` function to use your own data
- Update colors and styling in `styles.css` to match your brand

## Browser Support

Works in all modern browsers (Chrome, Firefox, Safari, Edge).