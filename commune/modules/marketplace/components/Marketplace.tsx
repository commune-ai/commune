
```tsx
'use client';
import { useState, useEffect } from 'react';

interface Repo {
  id: number;
  name: string;
  description: string;
  html_url: string;
}

export default function Marketplace() {
  const [repos, setRepos] = useState<Repo[]>([]);

  useEffect(() => {
    // Example: Fetch trending repos from GitHub
    fetch('https://api.github.com/search/repositories?q=stars:>1&sort=stars')
      .then(res => res.json())
      .then(data => setRepos(data.items.slice(0, 5)));
  }, []);

  return (
    <div className="mt-8 text-white">
      <h2 className="text-xl mb-4">Marketplace</h2>
      <div className="space-y-4">
        {repos.map(repo => (
          <div key={repo.id} className="bg-gray-700 p-4 rounded">
            <h3 className="font-bold">{repo.name}</h3>
            <p className="text-sm text-gray-300">{repo.description}</p>
            <a 
              href={repo.html_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 text-sm"
            >
              View Repository
            </a>
          </div>
        ))}
      </div>
    </div>
  );
}
```
