
```tsx
interface FileExplorerProps {
  files: {[key: string]: string};
  setFiles: (files: {[key: string]: string}) => void;
  currentFile: string;
  setCurrentFile: (file: string) => void;
}

export default function FileExplorer({
  files,
  setFiles,
  currentFile,
  setCurrentFile
}: FileExplorerProps) {
  const createNewFile = () => {
    const filename = prompt('Enter file name:');
    if (filename) {
      setFiles(prev => ({
        ...prev,
        [filename]: ''
      }));
      setCurrentFile(filename);
    }
  };

  return (
    <div className="text-white">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl">Files</h2>
        <button 
          onClick={createNewFile}
          className="bg-blue-500 px-2 py-1 rounded"
        >
          New File
        </button>
      </div>
      <ul>
        {Object.keys(files).map(filename => (
          <li 
            key={filename}
            className={`cursor-pointer p-2 ${
              currentFile === filename ? 'bg-gray-700' : ''
            }`}
            onClick={() => setCurrentFile(filename)}
          >
            {filename}
          </li>
        ))}
      </ul>
    </div>
  );
}
```
