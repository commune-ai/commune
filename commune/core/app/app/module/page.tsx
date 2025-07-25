'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import Client from '@/lib/client';
import c from '@/lib/c';
import { ChevronRightIcon, ServerIcon, CodeBracketIcon, DocumentTextIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import ModuleCode from './ModuleCode';
import ModuleSchema from './ModuleSchema';

// Utility functions
const shorten = (s: string, max = 12) => {
  if (!s || s.length <= max) return s;
  return `${s.slice(0, 8)}...${s.slice(-4)}`;
};

const time2str = (timestamp: number) => {
  return new Date(timestamp * 1000).toLocaleString();
};

// Generate consistent color from module name
const text2color = (text: string) => {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = text.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 70%, 50%)`;
};

// Generate cyberpunk pattern
const generateCyberpunkPattern = (key: string, color: string) => {
  const svg = `
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="${color}" stroke-width="0.5" opacity="0.3"/>
        </pattern>
      </defs>
      <rect width="100" height="100" fill="url(#grid)" />
      <circle cx="10" cy="10" r="1" fill="${color}" opacity="0.6"/>
      <circle cx="30" cy="30" r="1" fill="${color}" opacity="0.6"/>
      <circle cx="50" cy="50" r="1" fill="${color}" opacity="0.6"/>
      <circle cx="70" cy="70" r="1" fill="${color}" opacity="0.6"/>
      <circle cx="90" cy="90" r="1" fill="${color}" opacity="0.6"/>
    </svg>
  `;
  return `data:image/svg+xml;base64,${btoa(svg)}`;
};

interface InfoCardProps {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
  color?: string;
  copyable?: boolean;
}

const InfoCard: React.FC<InfoCardProps> = ({ label, value, icon, color, copyable }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (copyable && value) {
      navigator.clipboard.writeText(String(value));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div 
      className="relative bg-black/40 backdrop-blur-sm border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-all duration-300 group overflow-hidden"
      onClick={copyable ? handleCopy : undefined}
      style={{ cursor: copyable ? 'pointer' : 'default' }}
    >
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-5" style={{ backgroundImage: `url(${generateCyberpunkPattern(label, color || '#00ff00')})` }} />
      
      {/* Glow effect on hover */}
      <div 
        className="absolute inset-0 opacity-0 group-hover:opacity-20 transition-opacity duration-300"
        style={{ 
          background: `radial-gradient(circle at center, ${color || '#00ff00'} 0%, transparent 70%)`,
        }}
      />
      
      <div className="relative z-10">
        <div className="flex items-center gap-2 mb-2">
          {icon && <div style={{ color }}>{icon}</div>}
          <span className="text-gray-400 text-sm uppercase tracking-wider">{label}</span>
        </div>
        <div className="font-mono text-lg text-white break-all">
          {value || 'N/A'}
        </div>
        {copyable && (
          <div className="absolute top-2 right-2 text-xs text-gray-500">
            {copied ? 'Copied!' : 'Click to copy'}
          </div>
        )}
      </div>
    </div>
  );
};

export default function ModuleClient({ params }: { params: { module_name: string } }) {
  const searchParams = useSearchParams();
  const [module, setModule] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'code' | 'schema'>('code');
  const [syncing, setSyncing] = useState(false);

  const moduleColor = text2color(params.module_name);

  const fetchModule = useCallback(async (update = false) => {
    try {
      setLoading(true);
      setError(null);
      const client = new Client();
      const data = await client.call('module/get_module', { name: params.module_name, update });
      setModule(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch module');
    } finally {
      setLoading(false);
      setSyncing(false);
    }
  }, [params.module_name]);

  useEffect(() => {
    fetchModule();
  }, [fetchModule]);

  const handleSync = async () => {
    setSyncing(true);
    await fetchModule(true);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <div className="relative w-32 h-32 mx-auto mb-8">
            <div className="absolute inset-0 border-4 border-t-transparent rounded-full animate-spin" style={{ borderColor: moduleColor }} />
            <div className="absolute inset-4 border-4 border-b-transparent rounded-full animate-spin-reverse" style={{ borderColor: moduleColor }} />
          </div>
          <p className="text-gray-400 animate-pulse">Loading module...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center p-4">
        <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-8 max-w-md w-full">
          <h2 className="text-red-400 text-xl font-bold mb-4">Error Loading Module</h2>
          <p className="text-gray-300 mb-6">{error}</p>
          <button
            onClick={() => fetchModule()}
            className="w-full bg-red-500/20 hover:bg-red-500/30 text-red-400 py-2 px-4 rounded-lg transition-colors duration-200"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Background with module-specific color */}
      <div className="fixed inset-0 opacity-10" style={{ 
        background: `radial-gradient(circle at 20% 50%, ${moduleColor} 0%, transparent 50%),
                     radial-gradient(circle at 80% 80%, ${moduleColor} 0%, transparent 50%)` 
      }} />
      
      <div className="relative z-10">
        {/* Header */}
        <div className="border-b border-gray-800 bg-black/50 backdrop-blur-sm sticky top-0 z-20">
          <div className="container mx-auto px-4 py-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-3 rounded-lg" style={{ backgroundColor: `${moduleColor}20`, border: `1px solid ${moduleColor}40` }}>
                  <ServerIcon className="w-6 h-6" style={{ color: moduleColor }} />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white">{params.module_name}</h1>
                  <p className="text-gray-400 text-sm">Module Details</p>
                </div>
              </div>
              <button
                onClick={handleSync}
                disabled={syncing}
                className="flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 hover:scale-105"
                style={{ 
                  backgroundColor: `${moduleColor}20`,
                  border: `1px solid ${moduleColor}40`,
                  color: moduleColor
                }}
              >
                <ArrowPathIcon className={`w-4 h-4 ${syncing ? 'animate-spin' : ''}`} />
                <span>{syncing ? 'Syncing...' : 'Sync'}</span>
              </button>
            </div>
          </div>
        </div>

        {/* Module Info Grid */}
        <div className="container mx-auto px-4 py-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <InfoCard 
              label="Address" 
              value={module?.address || 'N/A'} 
              color={moduleColor}
              copyable
            />
            <InfoCard 
              label="Stake" 
              value={module?.stake ? `${c.round(module.stake, 2)} COM` : 'N/A'} 
              color={moduleColor}
            />
            <InfoCard 
              label="Emission" 
              value={module?.emission ? `${c.round(module.emission, 4)} COM/day` : 'N/A'} 
              color={moduleColor}
            />
            <InfoCard 
              label="Last Update" 
              value={module?.last_update ? time2str(module.last_update) : 'N/A'} 
              color={moduleColor}
            />
          </div>

          {/* Tabs */}
          <div className="flex gap-1 mb-6 bg-black/40 backdrop-blur-sm rounded-lg p-1 w-fit">
            <button
              onClick={() => setActiveTab('code')}
              className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all duration-200 ${
                activeTab === 'code' 
                  ? 'text-white' 
                  : 'bg-transparent text-gray-400 hover:text-gray-200'
              }`}
              style={{
                backgroundColor: activeTab === 'code' ? `${moduleColor}30` : 'transparent',
                borderColor: activeTab === 'code' ? moduleColor : 'transparent',
                borderWidth: '1px',
                borderStyle: 'solid'
              }}
            >
              <CodeBracketIcon className="w-4 h-4" />
              <span>Code</span>
            </button>
            <button
              onClick={() => setActiveTab('schema')}
              className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all duration-200 ${
                activeTab === 'schema' 
                  ? 'text-white' 
                  : 'bg-transparent text-gray-400 hover:text-gray-200'
              }`}
              style={{
                backgroundColor: activeTab === 'schema' ? `${moduleColor}30` : 'transparent',
                borderColor: activeTab === 'schema' ? moduleColor : 'transparent',
                borderWidth: '1px',
                borderStyle: 'solid'
              }}
            >
              <DocumentTextIcon className="w-4 h-4" />
              <span>API Schema</span>
            </button>
          </div>

          {/* Tab Content */}
          <div className="bg-black/40 backdrop-blur-sm border border-gray-800 rounded-lg overflow-hidden">
            {activeTab === 'code' && module?.code && (
              <ModuleCode 
                files={module.code} 
                title={`${params.module_name} Source Code`}
                showFileTree={searchParams.get('file_tree') !== 'false'}
                showSearch={searchParams.get('search') !== 'false'}
              />
            )}
            {activeTab === 'schema' && module?.schema && (
              <ModuleSchema schema={module.schema} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}