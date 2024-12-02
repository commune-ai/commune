
```tsx
'use client';
import { useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { dracula } from '@uiw/codemirror-theme-dracula';

interface EditorProps {
  content: string;
  onChange: (value: string) => void;
}

export default function Editor({ content, onChange }: EditorProps) {
  return (
    <CodeMirror
      value={content}
      height="100vh"
      theme={dracula}
      extensions={[javascript()]}
      onChange={onChange}
    />
  );
}
```
