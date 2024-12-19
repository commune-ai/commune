
```tsx
'use client';
import { useState } from 'react';
import Editor from '@/components/Editor';
import FileExplorer from '@/components/FileExplorer';
import ThemePicker from '@/components/ThemePicker';
import Marketplace from '@/components/Marketplace';

export default function Home() {
  const [currentFile, setCurrentFile] = useState('');
  const [files, setFiles] = useState<{[key: string]: string}>({});
  
  return (
    <div className="flex h-screen">
      <aside className="w-64 bg-gray-800 p-4">
        <FileExplorer 
          files={files}
          setFiles={setFiles}
          currentFile={currentFile}
          setCurrentFile={setCurrentFile}
        />
        <ThemePicker />
        <Marketplace />
      </aside>
      <main className="flex-1">
        <Editor 
          content={files[currentFile] || ''}
          onChange={(content) => {
            setFiles(prev => ({
              ...prev,
              [currentFile]: content
            }));
          }}
        />
      </main>
    </div>
  );
}
```
