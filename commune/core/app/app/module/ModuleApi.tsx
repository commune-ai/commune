'use client';

import { useState, useMemo, useEffect, useCallback } from 'react';
import { Client } from '@/app/client/client';
import { CopyButton } from '@/app/components/CopyButton';
import { useUserContext } from '@/app/context/UserContext';
import { Auth } from '@/app/key';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MagnifyingGlassIcon,
  CodeBracketIcon,
  PlayIcon,
  CommandLineIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

type SchemaField = { value: any; type: string };
type SchemaType = {
  input: Record<string, SchemaField>;
  output: SchemaField;
  code?: string;
  hash?: string;
};
type TabType = 'run' | 'code';

// Minimal, neutral, “Apple-pro” palette (dark)
const ui = {
  bg:       '#0b0b0b',
  panel:    '#121212',
  panelAlt: '#151515',
  border:   '#2a2a2a',
  text:     '#e7e7e7',
  textDim:  '#a8a8a8',
  focus:    '#3a86ff',
  accent:   '#ffffff',
  danger:   '#ff3b30',
};

export const ModuleSchema = ({ mod }: { mod: any }) => {
  const { keyInstance } = useUserContext();

  const schema: Record<string, SchemaType> = mod?.schema || {};

  // filter out self/cls (memoized)
  const filteredSchema = useMemo(() => {
    return Object.entries(schema).reduce((acc, [fn, value]) => {
      if (fn === 'self' || fn === 'cls') return acc;
      const filteredInput = Object.entries(value?.input || {}).reduce((ia, [k, v]) => {
        if (k !== 'self' && k !== 'cls') (ia as any)[k] = v;
        return ia;
      }, {} as Record<string, SchemaField>);
      (acc as any)[fn] = { ...value, input: filteredInput };
      return acc;
    }, {} as Record<string, SchemaType>);
  }, [schema]);

  const functionNames = useMemo(() => Object.keys(filteredSchema), [filteredSchema]);

  const [selectedFunction, setSelectedFunction] = useState<string>('');
  const [params, setParams] = useState<Record<string, any>>({});
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [authHeaders, setAuthHeaders] = useState<any>(null);
  const [urlParams, setUrlParams] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [activeTab, setActiveTab] = useState<TabType>('run');

  // init selection (avoid setting state during render)
  useEffect(() => {
    if (!selectedFunction && functionNames.length > 0) {
      const first = functionNames[0];
      setSelectedFunction(first);
      const d: Record<string, any> = {};
      const inp = filteredSchema[first]?.input || {};
      Object.entries(inp).forEach(([p, det]) => {
        if (det.value !== '_empty' && det.value !== undefined) d[p] = det.value;
      });
      setParams(d);
    }
  }, [selectedFunction, functionNames, filteredSchema]);

  const searchedFunctions = useMemo(() => {
    if (!searchTerm) return functionNames;
    const q = searchTerm.toLowerCase();
    return functionNames.filter((fn) => fn.toLowerCase().includes(q));
  }, [functionNames, searchTerm]);

  const handleParamChange = (p: string, v: string) =>
    setParams((prev) => ({ ...prev, [p]: v }));

  const initializeParams = useCallback((fn: string) => {
    const s = filteredSchema[fn];
    const d: Record<string, any> = {};
    Object.entries(s?.input || {}).forEach(([p, det]) => {
      if (det.value !== '_empty' && det.value !== undefined) d[p] = det.value;
    });
    setParams(d);
  }, [filteredSchema]);

  const executeFunction = async () => {
    if (!selectedFunction) return;
    setLoading(true);
    setError('');
    setAuthHeaders(null);
    setResponse(null);

    try {
      const client = new Client(undefined, keyInstance);
      const auth = new Auth(keyInstance);
      const headers = auth.generate({ fn: selectedFunction, params });
      setAuthHeaders(headers);

      // build url params safely (stringify objects)
      const qs = new URLSearchParams(
        Object.fromEntries(
          Object.entries(params).map(([k, v]) => [
            k,
            typeof v === 'object' ? JSON.stringify(v) : String(v ?? ''),
          ])
        )
      ).toString();
      setUrlParams(qs);

      const res = await client.call('call', { fn: selectedFunction, params });
      setResponse(res);
    } catch (err: any) {
      setError(err?.message || 'Failed to execute function');
    } finally {
      setLoading(false);
    }
  };

  // subtle edge fade for scroll rows
  const fadeMaskX: React.CSSProperties = {
    WebkitMaskImage:
      'linear-gradient(to right, transparent, black 12px, black calc(100% - 12px), transparent)',
    maskImage:
      'linear-gradient(to right, transparent, black 12px, black calc(100% - 12px), transparent)',
  };

  return (
    <div className="flex h-full flex-col font-mono" style={{ backgroundColor: ui.bg }}>
      {/* Top bar */}
      <div className="border-b px-4 py-3" style={{ borderColor: ui.border, backgroundColor: ui.panel }}>
        <div className="relative">
          <input
            type="text"
            placeholder="Search functions"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full rounded-md px-3 py-2 text-sm outline-none"
            style={{
              backgroundColor: ui.panelAlt,
              color: ui.text,
              border: `1px solid ${ui.border}`,
              transition: 'box-shadow 120ms ease, border-color 120ms ease',
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = ui.focus;
              e.currentTarget.style.boxShadow = `0 0 0 3px ${ui.focus}22`;
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = ui.border;
              e.currentTarget.style.boxShadow = 'none';
            }}
            onKeyDown={(e) => {
              if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') executeFunction();
            }}
          />
          <MagnifyingGlassIcon
            className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 opacity-60"
            style={{ color: ui.textDim }}
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute left-3 top-1/2 -translate-y-1/2"
              aria-label="Clear"
            >
              <XMarkIcon className="h-4 w-4" style={{ color: ui.textDim }} />
            </button>
          )}
        </div>
      </div>

      {/* Function chips (horizontal scroll w/ subtle scrollbar + edge fade) */}
      <div
        className="micro-scroll -mx-4 flex max-w-full items-center gap-1.5 overflow-x-auto px-4 py-2"
        style={fadeMaskX}
      >
        {searchedFunctions.map((fn) => {
          const active = selectedFunction === fn;
          return (
            <motion.button
              key={fn}
              onClick={() => {
                setSelectedFunction(fn);
                initializeParams(fn);
                setResponse(null);
                setError('');
                setAuthHeaders(null);
                setUrlParams('');
                setActiveTab('run');
              }}
              className="whitespace-nowrap rounded-md px-3 py-1.5 text-xs"
              style={{
                backgroundColor: active ? ui.accent : ui.panelAlt,
                color: active ? '#000' : ui.text,
                border: `1px solid ${active ? ui.accent : ui.border}`,
                transition: 'background-color 120ms ease, border-color 120ms ease',
              }}
              whileTap={{ scale: 0.98 }}
              title={fn}
            >
              {fn}
            </motion.button>
          );
        })}
        {searchedFunctions.length === 0 && (
          <span className="text-xs" style={{ color: ui.textDim }}>
            No functions found
          </span>
        )}
      </div>

      {/* Main content */}
      <div className="flex-1 px-4 pb-4">
        {selectedFunction ? (
          <div
            className="flex h-full flex-col rounded-lg"
            style={{
              backgroundColor: ui.panel,
              border: `1px solid ${ui.border}`,
              boxShadow: '0 4px 20px rgba(0,0,0,0.35)',
            }}
          >
            {/* Tabs */}
            <div className="flex gap-1 border-b p-2" style={{ borderColor: ui.border, backgroundColor: ui.panel }}>
              <button
                onClick={() => setActiveTab('run')}
                className="rounded-md px-3 py-1.5 text-xs"
                style={{
                  backgroundColor: activeTab === 'run' ? ui.panelAlt : 'transparent',
                  color: ui.text,
                  border: `1px solid ${ui.border}`,
                }}
              >
                <span className="inline-flex items-center gap-1">
                  <PlayIcon className="h-3.5 w-3.5" />
                  Run
                </span>
              </button>
              {filteredSchema[selectedFunction]?.code && (
                <button
                  onClick={() => setActiveTab('code')}
                  className="rounded-md px-3 py-1.5 text-xs"
                  style={{
                    backgroundColor: activeTab === 'code' ? ui.panelAlt : 'transparent',
                    color: ui.text,
                    border: `1px solid ${ui.border}`,
                  }}
                >
                  <span className="inline-flex items-center gap-1">
                    <CodeBracketIcon className="h-3.5 w-3.5" />
                    Code
                  </span>
                </button>
              )}
            </div>

            {/* Tab content */}
            <div className="flex-1 overflow-auto">
              <AnimatePresence mode="wait">
                {activeTab === 'run' ? (
                  <motion.div
                    key="run"
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    className="space-y-4 p-4"
                  >
                    {/* Parameters */}
                    <div className="space-y-3">
                      {Object.entries(filteredSchema[selectedFunction]?.input || {}).map(([p, d]) => (
                        <div key={p} className="grid gap-2 md:grid-cols-3">
                          <label className="text-xs md:col-span-1" style={{ color: ui.textDim }}>
                            {p} <span style={{ color: ui.textDim }}>[{d.type}]</span>
                          </label>
                          <input
                            type="text"
                            value={params[p] ?? ''}
                            onChange={(e) => handleParamChange(p, e.target.value)}
                            placeholder={d.value ? String(d.value) : ''}
                            className="rounded-md px-3 py-2 text-xs outline-none md:col-span-2"
                            style={{
                              backgroundColor: ui.panelAlt,
                              color: ui.text,
                              border: `1px solid ${ui.border}`,
                              transition: 'box-shadow 120ms ease, border-color 120ms ease',
                            }}
                            onFocus={(e) => {
                              e.currentTarget.style.borderColor = ui.focus;
                              e.currentTarget.style.boxShadow = `0 0 0 3px ${ui.focus}22`;
                            }}
                            onBlur={(e) => {
                              e.currentTarget.style.borderColor = ui.border;
                              e.currentTarget.style.boxShadow = 'none';
                            }}
                            onKeyDown={(e) => {
                              if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') executeFunction();
                            }}
                          />
                        </div>
                      ))}
                    </div>

                    {/* Run */}
                    <div className="pt-2">
                      <button
                        onClick={executeFunction}
                        disabled={loading}
                        className="w-full rounded-md px-4 py-2 text-sm font-semibold outline-none"
                        style={{
                          backgroundColor: ui.accent,
                          color: '#000',
                          opacity: loading ? 0.7 : 1,
                          border: `1px solid ${ui.accent}`,
                        }}
                        title="Ctrl/⌘ + Enter to Run"
                      >
                        {loading ? 'Running…' : 'Run'}
                      </button>
                    </div>

                    {/* Output */}
                    {(response || error) && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold" style={{ color: error ? ui.danger : ui.text }}>
                            {error ? 'Error' : 'Response'}
                          </span>
                          <CopyButton content={JSON.stringify(response || error, null, 2)} />
                        </div>
                        <pre
                          className="micro-scroll-y max-h-64 overflow-auto rounded-md p-3 text-xs"
                          style={{
                            backgroundColor: ui.panelAlt,
                            color: error ? ui.danger : ui.text,
                            border: `1px solid ${error ? ui.danger : ui.border}`,
                          }}
                        >
{JSON.stringify(response || error, null, 2)}
                        </pre>
                      </div>
                    )}
                  </motion.div>
                ) : (
                  <motion.div
                    key="code"
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    className="h-full p-4"
                  >
                    <div
                      className="micro-scroll-y h-full overflow-auto rounded-md"
                      style={{ backgroundColor: ui.panelAlt, border: `1px solid ${ui.border}` }}
                    >
                      <div
                        className="flex items-center justify-between border-b px-3 py-2"
                        style={{ borderColor: ui.border }}
                      >
                        <span className="text-xs" style={{ color: ui.textDim }}>
                          Function source
                        </span>
                        <CopyButton content={filteredSchema[selectedFunction]?.code || ''} />
                      </div>
                      <pre className="p-3 text-xs" style={{ color: ui.text }}>
                        {filteredSchema[selectedFunction]?.code || 'No code available'}
                      </pre>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        ) : (
          <div
            className="flex h-full items-center justify-center rounded-lg border"
            style={{ borderColor: ui.border, color: ui.textDim, backgroundColor: ui.panel }}
          >
            <div className="flex items-center gap-2 text-sm">
              <CommandLineIcon className="h-5 w-5 opacity-60" />
              Select a function to begin
            </div>
          </div>
        )}
      </div>

      {/* subtle, hover-only scrollbar styling */}
      <style jsx>{`
        .micro-scroll {
          scrollbar-width: thin;
          scrollbar-color: transparent transparent;
        }
        .micro-scroll:hover {
          scrollbar-color: rgba(255,255,255,0.14) transparent;
        }
        .micro-scroll::-webkit-scrollbar {
          height: 6px;
          background: transparent;
        }
        .micro-scroll::-webkit-scrollbar-thumb {
          background: transparent;
          border-radius: 9999px;
        }
        .micro-scroll:hover::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,0.12);
        }
        .micro-scroll::-webkit-scrollbar-track {
          background: transparent;
        }

        .micro-scroll-y {
          scrollbar-width: thin;
          scrollbar-color: rgba(255,255,255,0.12) transparent;
        }
        .micro-scroll-y::-webkit-scrollbar {
          width: 8px;
          background: transparent;
        }
        .micro-scroll-y::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,0.12);
          border-radius: 9999px;
        }
        .micro-scroll-y::-webkit-scrollbar-track {
          background: transparent;
        }
      `}</style>
    </div>
  );
};

export default ModuleSchema;