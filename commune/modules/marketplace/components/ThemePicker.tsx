
```tsx
'use client';
import { useState } from 'react';

const themes = [
  { name: 'Dracula', value: 'dracula' },
  { name: 'GitHub', value: 'github' },
  { name: 'VS Code Dark', value: 'vscode-dark' },
];

export default function ThemePicker() {
  const [currentTheme, setCurrentTheme] = useState('dracula');

  return (
    <div className="mt-8 text-white">
      <h2 className="text-xl mb-4">Themes</h2>
      <select 
        value={currentTheme}
        onChange={(e) => setCurrentTheme(e.target.value)}
        className="bg-gray-700 p-2 rounded w-full"
      >
        {themes.map(theme => (
          <option key={theme.value} value={theme.value}>
            {theme.name}
          </option>
        ))}
      </select>
    </div>
  );
}
```
