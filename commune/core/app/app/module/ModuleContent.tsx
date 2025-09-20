'use client';

import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { CopyButton } from '@/app/components/CopyButton';
import {
  ChevronDownIcon, ChevronRightIcon, DocumentIcon, FolderIcon, FolderOpenIcon,
  MagnifyingGlassIcon, CodeBracketIcon, DocumentTextIcon, PhotoIcon, FilmIcon,
  MusicalNoteIcon, ArchiveBoxIcon, DocumentChartBarIcon, ClipboardDocumentIcon
} from '@heroicons/react/24/outline';

interface ModuleContentProps {
  files: Record<string, string>;
  title?: string;
  showSearch?: boolean;
  showFileTree?: boolean;
  compactMode?: boolean;
  defaultExpandedFolders?: boolean;
}

type FileNode = {
  name: string;
  path: string;
  type: 'file' | 'folder';
  children?: FileNode[];
  content?: string;
  language?: string;
  hash?: string;
  lineCount?: number;
  size?: string;
};

const ui = {
  panel: '#0b0b0b',
  panelAlt: '#121212',
  panelAlt2: '#151515',
  border: '#2a2a2a',
  text: '#e7e7e7',
  textDim: '#a8a8a8',
  green: '#22c55e',
  yellow: '#facc15',
};

const getFileIcon = (filename: string) => {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  const iconMap: Record<string, any> = {
    ts: CodeBracketIcon, tsx: CodeBracketIcon, js: CodeBracketIcon, jsx: CodeBracketIcon, py: CodeBracketIcon,
    json: DocumentChartBarIcon, css: DocumentTextIcon, html: DocumentTextIcon, md: DocumentTextIcon, txt: DocumentTextIcon,
    jpg: PhotoIcon, jpeg: PhotoIcon, png: PhotoIcon, gif: PhotoIcon, svg: PhotoIcon,
    mp4: FilmIcon, avi: FilmIcon, mov: FilmIcon, mp3: MusicalNoteIcon, wav: MusicalNoteIcon, zip: ArchiveBoxIcon, tar: ArchiveBoxIcon, gz: ArchiveBoxIcon,
  };
  return iconMap[ext] || DocumentIcon;
};

const getLanguageFromPath = (path: string): string => {
  const ext = path.split('.').pop()?.toLowerCase() || '';
  const langMap: Record<string, string> = {
    ts: 'typescript', tsx: 'typescript', js: 'javascript', jsx: 'javascript',
    py: 'python', json: 'json', css: 'css', html: 'html', md: 'markdown',
  };
  return langMap[ext] || 'text';
};

const languageColors: Record<string, string> = {
  typescript: 'text-blue-400', javascript: 'text-yellow-400', python: 'text-green-400',
  json: 'text-orange-400', css: 'text-pink-400', html: 'text-red-400',
  markdown: 'text-gray-400', text: 'text-gray-300',
};

const formatFileSize = (bytes: number): string =>
  bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`;

/** fast, browser-safe 32-bit FNV-1a hash → 8-hex */
const hashShort = (str: string): string => {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24); }
  return ('0000000' + (h >>> 0).toString(16)).slice(-8);
};

/** build sorted file tree (folders first, A→Z) */
const buildFileTree = (files: Record<string, string>): FileNode[] => {
  const root: FileNode = { name: '', path: '', type: 'folder', children: [] };

  Object.entries(files).forEach(([path, content]) => {
    const parts = path.split('/').filter(Boolean);
    let current = root;
    parts.forEach((part, idx) => {
      const isFile = idx === parts.length - 1;
      const currentPath = parts.slice(0, idx + 1).join('/');
      let child = current.children!.find((c) => c.name === part);
      if (!child) {
        child = {
          name: part,
          path: currentPath,
          type: isFile ? 'file' : 'folder',
          children: isFile ? undefined : [],
          content: isFile ? content : undefined,
          language: isFile ? getLanguageFromPath(part) : undefined,
          hash: isFile ? hashShort(content) : undefined,
          lineCount: isFile ? content.split('\n').length : undefined,
          size: isFile ? formatFileSize(content.length) : undefined,
        };
        current.children!.push(child);
      }
      if (!isFile) current = child;
    });
  });

  const sortNodes = (nodes?: FileNode[]) => {
    if (!nodes) return;
    nodes.sort((a, b) => (a.type === b.type ? a.name.localeCompare(b.name) : a.type === 'folder' ? -1 : 1));
    nodes.forEach((n) => sortNodes(n.children));
  };
  sortNodes(root.children);
  return root.children || [];
};

/** safer highlighter (no regex.lastIndex bug) */
const highlightSearchTerm = (text: string, term: string) => {
  if (!term) return text;
  const safe = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const parts = text.split(new RegExp(`(${safe})`, 'gi'));
  return (
    <>
      {parts.map((p, i) =>
        i % 2 === 1 ? (
          <span key={i} className="bg-yellow-400/30 text-yellow-300 font-bold">{p}</span>
        ) : (
          <span key={i}>{p}</span>
        )
      )}
    </>
  );
};

function FileTreeItem({
  node, level, onSelect, expandedFolders, toggleFolder, selectedPath, onCopy, searchTerm,
}: {
  node: FileNode; level: number; onSelect: (n: FileNode) => void;
  expandedFolders: Set<string>; toggleFolder: (p: string) => void; selectedPath?: string;
  onCopy: (n: FileNode) => void; searchTerm?: string;
}) {
  const isExpanded = expandedFolders.has(node.path);
  const isSelected = selectedPath === node.path;
  const FileIcon = node.type === 'file' ? getFileIcon(node.name) : (isExpanded ? FolderOpenIcon : FolderIcon);

  const matchesSearch = searchTerm
    ? node.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      node.path.toLowerCase().includes(searchTerm.toLowerCase())
    : true;

  const handleClick = () => (node.type === 'folder' ? toggleFolder(node.path) : onSelect(node));
  if (!matchesSearch && node.type === 'file') return null;

  return (
    <div>
      <div
        className={`group micro-row flex cursor-pointer items-center rounded-md px-2 py-1.5 text-xs transition-all duration-150
        ${isSelected ? 'bg-emerald-900/25 text-emerald-300' : 'text-gray-400'}
        ${matchesSearch && searchTerm ? 'ring-1 ring-yellow-400/30' : ''}`}
        style={{ paddingLeft: `${level * 12 + 8}px` }}
        onClick={handleClick}
        title={node.path}
      >
        {node.type === 'folder' ? (
          isExpanded ? <ChevronDownIcon className="mr-1 h-3 w-3" /> : <ChevronRightIcon className="mr-1 h-3 w-3" />
        ) : null}
        <FileIcon className={`mr-2 h-4 w-4 flex-shrink-0 ${node.type === 'folder' ? 'text-yellow-500' : 'text-gray-400'}`} />
        <span className="flex-1 truncate font-mono">
          {searchTerm ? highlightSearchTerm(node.name, searchTerm) : node.name}
        </span>
        {node.type === 'file' ? (
          <>
            <span className="ml-2 text-xs opacity-60">{node.size}</span>
            <button
              onClick={(e) => { e.stopPropagation(); onCopy(node); }}
              className="ml-2 opacity-0 transition-opacity group-hover:opacity-100"
              title="Copy file content"
            >
              <ClipboardDocumentIcon className="h-3 w-3 text-emerald-400 hover:text-emerald-300" />
            </button>
          </>
        ) : (
          <button
            onClick={(e) => { e.stopPropagation(); onCopy(node); }}
            className="ml-2 opacity-0 transition-opacity group-hover:opacity-100"
            title="Copy folder contents"
          >
            <ClipboardDocumentIcon className="h-3 w-3 text-emerald-400 hover:text-emerald-300" />
          </button>
        )}
      </div>
      {node.type === 'folder' && isExpanded && node.children && (
        <div>
          {node.children.map((child) => (
            <FileTreeItem
              key={child.path}
              node={child}
              level={level + 1}
              onSelect={onSelect}
              expandedFolders={expandedFolders}
              toggleFolder={toggleFolder}
              selectedPath={selectedPath}
              onCopy={onCopy}
              searchTerm={searchTerm}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export const ModuleContent: React.FC<ModuleContentProps> = ({
  files,
  showSearch = true,
  showFileTree = true,
  compactMode = false,
  defaultExpandedFolders = true,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [fileSearchTerm, setFileSearchTerm] = useState('');
  const [collapsedFiles, setCollapsedFiles] = useState<Set<string>>(new Set());
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [searchResults, setSearchResults] = useState<{ path: string; lineNumbers: number[] }[]>([]);
  const codeRefs = useRef<Record<string, HTMLDivElement | null>>({});

  // build tree (sorted)
  useEffect(() => {
    const tree = buildFileTree(files);
    setFileTree(tree);

    if (defaultExpandedFolders) {
      const all = new Set<string>();
      const collect = (nodes: FileNode[]) => nodes.forEach((n) => { if (n.type === 'folder') { all.add(n.path); n.children && collect(n.children); }});
      collect(tree);
      setExpandedFolders(all);
    }
  }, [files, defaultExpandedFolders]);

  // expand folders for file filter
  useEffect(() => {
    if (!fileSearchTerm) return;
    const folders = new Set<string>();
    const check = (n: FileNode, parent = '') => {
      const cp = parent ? `${parent}/${n.name}` : n.name;
      if (n.name.toLowerCase().includes(fileSearchTerm.toLowerCase()) || n.path.toLowerCase().includes(fileSearchTerm.toLowerCase())) {
        const parts = cp.split('/').filter(Boolean);
        for (let i = 0; i < parts.length - 1; i++) folders.add(parts.slice(0, i + 1).join('/'));
      }
      n.children?.forEach((c) => check(c, cp));
    };
    fileTree.forEach((n) => check(n));
    setExpandedFolders((prev) => new Set([...prev, ...folders]));
  }, [fileSearchTerm, fileTree]);

  // search in file contents
  useEffect(() => {
    if (!searchTerm) { setSearchResults([]); return; }

    const results: { path: string; lineNumbers: number[] }[] = [];
    Object.entries(files).forEach(([path, content]) => {
      const lines = content.split('\n');
      const matchLines: number[] = [];
      const q = searchTerm.toLowerCase();
      lines.forEach((line, idx) => { if (line.toLowerCase().includes(q)) matchLines.push(idx + 1); });
      if (matchLines.length) results.push({ path, lineNumbers: matchLines });
    });
    setSearchResults(results);
    setCollapsedFiles(new Set()); // expand files with matches
  }, [searchTerm, files]);

  const fileSections = useMemo(() =>
    Object.entries(files).map(([path, content]) => ({
      path,
      name: path.split('/').pop() || path,
      content,
      language: getLanguageFromPath(path),
      hash: hashShort(content),
      lineCount: content.split('\n').length,
      size: formatFileSize(content.length),
    })), [files]
  );

  const filteredSections = useMemo(() => {
    if (!searchTerm) return fileSections;
    const q = searchTerm.toLowerCase();
    return fileSections.filter((s) => s.path.toLowerCase().includes(q) || s.content.toLowerCase().includes(q));
  }, [fileSections, searchTerm]);

  const stats = useMemo(() => {
    const totalLines = filteredSections.reduce((sum, s) => sum + s.lineCount, 0);
    const totalSize = filteredSections.reduce((sum, s) => sum + s.content.length, 0);
    return { fileCount: filteredSections.length, totalLines, totalSize: formatFileSize(totalSize) };
  }, [filteredSections]);

  const toggleFile = (path: string) =>
    setCollapsedFiles((prev) => { const n = new Set(prev); n.has(path) ? n.delete(path) : n.add(path); return n; });

  const toggleFolder = (path: string) =>
    setExpandedFolders((prev) => { const n = new Set(prev); n.has(path) ? n.delete(path) : n.add(path); return n; });

  const handleFileSelect = (node: FileNode) => {
    if (node.type !== 'file') return;
    setSelectedFile(node.path);
    const el = codeRefs.current[node.path];
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const copyFileContent = (node: FileNode) => {
    if (node.type === 'file' && node.content) {
      navigator.clipboard.writeText(node.content);
      return;
    }
    if (node.type === 'folder') {
      const buffer: string[] = [];
      const walk = (n: FileNode) => {
        if (n.type === 'file' && n.content) buffer.push(`// ${n.path}\n${n.content}`);
        n.children?.forEach(walk);
      };
      walk(node);
      navigator.clipboard.writeText(buffer.join('\n\n'));
    }
  };

  const renderLineNumbers = (content: string, startLine: number, path: string) => {
    const lines = content.split('\n');
    const matches = searchResults.find((r) => r.path === path)?.lineNumbers || [];
    return (
      <div className="select-none pr-2 font-mono text-xs text-gray-500">
        {lines.map((_, i) => {
          const ln = startLine + i;
          const isMatch = matches.includes(ln);
          return (
            <div key={i} className={`text-right ${isMatch ? 'bg-yellow-400/20 text-yellow-400' : ''}`}>
              {ln}
            </div>
          );
        })}
      </div>
    );
  };

  const renderCode = (content: string, language: string, path: string) => {
    const langColor = languageColors[language] || 'text-gray-300';
    const lines = content.split('\n');
    const matches = searchResults.find((r) => r.path === path)?.lineNumbers || [];
    return (
      <pre className="micro-scroll micro-edge-x flex-1 overflow-x-auto">
        <code className={`font-mono text-xs leading-relaxed ${langColor}`}>
          {lines.map((line, i) => {
            const ln = i + 1;
            const isMatch = matches.includes(ln);
            return (
              <div key={i} className={isMatch ? 'bg-yellow-400/20' : ''}>
                {searchTerm ? highlightSearchTerm(line, searchTerm) : line}
              </div>
            );
          })}
        </code>
      </pre>
    );
  };

  return (
    <div className="overflow-hidden rounded-lg" style={{ backgroundColor: ui.panelAlt }}>
      {/* Header */}
      <div className="px-4 py-3" style={{ backgroundColor: ui.panel, borderBottom: `1px solid ${ui.border}` }}>
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-4 text-xs" style={{ color: ui.textDim }}>
            <span>{stats.fileCount} files</span>
            <span>{stats.totalLines} lines</span>
            <span>{stats.totalSize}</span>
          </div>
        </div>

        {/* Search */}
        {showSearch && (
          <div className="relative">
            <MagnifyingGlassIcon className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search in code content…"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full rounded-md border bg-transparent pl-10 pr-16 py-2 text-sm outline-none"
              style={{ borderColor: ui.border, color: ui.text }}
            />
            {searchTerm && (
              <div className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs" style={{ color: ui.textDim }}>
                {searchResults.length} files, {searchResults.reduce((s, r) => s + r.lineNumbers.length, 0)} matches
              </div>
            )}
          </div>
        )}
      </div>

      {/* Body */}
      <div className="flex">
        {/* File tree */}
        {showFileTree && (
          <div className="micro-scroll-y w-64 max-h-[600px] overflow-y-auto border-r p-3"
               style={{ borderColor: ui.border, backgroundColor: ui.panelAlt2 }}>
            <div className="mb-2">
              <h3 className="mb-2 text-sm font-medium" style={{ color: ui.text }}>
                File Explorer
              </h3>
              <div className="relative">
                <MagnifyingGlassIcon className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  placeholder="Filter files…"
                  value={fileSearchTerm}
                  onChange={(e) => setFileSearchTerm(e.target.value)}
                  className="w-full rounded border bg-transparent pl-6 pr-2 py-1 text-xs outline-none"
                  style={{ borderColor: ui.border, color: ui.text }}
                />
              </div>
              <div className="mt-2 flex gap-1">
                <button
                  onClick={() => {
                    const all = new Set<string>();
                    const collect = (nodes: FileNode[]) =>
                      nodes.forEach((n) => { if (n.type === 'folder') { all.add(n.path); n.children && collect(n.children); }});
                    collect(fileTree);
                    setExpandedFolders(all);
                  }}
                  className="text-xs text-emerald-400 hover:opacity-80"
                  title="Expand all"
                >
                  <ChevronDownIcon className="h-3 w-3" />
                </button>
                <button
                  onClick={() => setExpandedFolders(new Set())}
                  className="text-xs text-emerald-400 hover:opacity-80"
                  title="Collapse all"
                >
                  <ChevronRightIcon className="h-3 w-3" />
                </button>
              </div>
            </div>

            <div className="space-y-0">
              {fileTree.map((node) => (
                <FileTreeItem
                  key={node.path}
                  node={node}
                  level={0}
                  onSelect={handleFileSelect}
                  expandedFolders={expandedFolders}
                  toggleFolder={toggleFolder}
                  selectedPath={selectedFile || undefined}
                  onCopy={copyFileContent}
                  searchTerm={fileSearchTerm}
                />
              ))}
            </div>
          </div>
        )}

        {/* Code view */}
        <div className="micro-scroll-y max-h-[600px] flex-1 overflow-y-auto">
          {filteredSections.map((section) => {
            const isCollapsed = collapsedFiles.has(section.path);
            const isSelected = selectedFile === section.path;
            const FileIcon = getFileIcon(section.name);
            const matches = searchResults.find((r) => r.path === section.path)?.lineNumbers.length || 0;

            // if a file is selected via tree, only show that
            if (showFileTree && selectedFile && !isSelected) return null;

            return (
              <div
                key={section.path}
                ref={(el) => { codeRefs.current[section.path] = el; }}
                className={`${isSelected ? 'bg-emerald-900/10' : ''} ${matches ? 'ring-1 ring-yellow-400/30' : ''}`}
              >
                {/* File header */}
                <div
                  className="flex cursor-pointer items-center justify-between bg-black/30 p-3 hover:bg-black/45"
                  onClick={() => toggleFile(section.path)}
                >
                  <div className="micro-row flex items-center gap-2">
                    {isCollapsed ? <ChevronRightIcon className="h-4 w-4 text-gray-400" /> : <ChevronDownIcon className="h-4 w-4 text-gray-400" />}
                    <FileIcon className="h-4 w-4 text-gray-400" />
                    <span className="font-mono text-sm text-emerald-400">
                      {searchTerm && section.path.toLowerCase().includes(searchTerm.toLowerCase())
                        ? highlightSearchTerm(section.path, searchTerm)
                        : section.path}
                    </span>
                    <span className={`text-xs ${languageColors[section.language]}`}>{section.language.toUpperCase()}</span>
                    {!!matches && <span className="text-xs text-yellow-400">{matches} matches</span>}
                  </div>
                  <div className="flex items-center gap-3 text-xs" style={{ color: ui.textDim }}>
                    <span>{section.size}</span>
                    <span>{section.lineCount} lines</span>
                    <span className="font-mono" style={{ color: ui.text }}>#{section.hash}</span>
                    <CopyButton content={section.content} />
                  </div>
                </div>

                {/* File content */}
                {!isCollapsed && (
                  <div className="flex" style={{ backgroundColor: '#0a0a0a' }}>
                    {!compactMode && renderLineNumbers(section.content, 1, section.path)}
                    <div className="micro-scroll micro-edge-x flex-1 overflow-x-auto p-3">
                      {renderCode(section.content, section.language, section.path)}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      {!compactMode && (
        <div className="flex items-center justify-between px-4 py-3"
             style={{ backgroundColor: ui.panel, borderTop: `1px solid ${ui.border}` }}>
          <div className="text-xs" style={{ color: ui.textDim }}>
            {searchTerm && searchResults.length > 0 && (
              <span>Found “{searchTerm}” in {searchResults.length} files</span>
            )}
          </div>
          <button
            onClick={() => {
              const all = filteredSections.map((s) => `// ${s.path}\n${s.content}`).join('\n\n');
              navigator.clipboard.writeText(all);
            }}
            className="flex items-center gap-2 text-xs text-emerald-400 hover:opacity-80"
          >
            <DocumentIcon className="h-3 w-3" />
            Copy All Code
          </button>
        </div>
      )}

      {/* subtle, hover-only scrollbars + edge-fade for horizontal areas */}
      <style jsx>{`
        .micro-scroll {
          scrollbar-width: thin;
          scrollbar-color: transparent transparent;
        }
        .micro-scroll:hover {
          scrollbar-color: rgba(255,255,255,0.14) transparent;
        }
        .micro-scroll::-webkit-scrollbar { height: 6px; background: transparent; }
        .micro-scroll::-webkit-scrollbar-thumb { background: transparent; border-radius: 9999px; }
        .micro-scroll:hover::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); }
        .micro-scroll::-webkit-scrollbar-track { background: transparent; }

        .micro-scroll-y {
          scrollbar-width: thin;
          scrollbar-color: rgba(255,255,255,0.12) transparent;
        }
        .micro-scroll-y::-webkit-scrollbar { width: 8px; background: transparent; }
        .micro-scroll-y::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 9999px; }
        .micro-scroll-y::-webkit-scrollbar-track { background: transparent; }

        .micro-edge-x {
          -webkit-mask-image: linear-gradient(to right, transparent, black 12px, black calc(100% - 12px), transparent);
                  mask-image: linear-gradient(to right, transparent, black 12px, black calc(100% - 12px), transparent);
        }

        .micro-row { min-height: 28px; }
      `}</style>
    </div>
  );
};

export default ModuleContent;
