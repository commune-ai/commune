'use client';

import { memo, useCallback, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { CopyButton } from '@/app/components/CopyButton';
import { ModuleType } from '@app/types/module';
import { User, Clock3, Tag as TagIcon, Boxes } from 'lucide-react';

interface ModuleCardProps { module: ModuleType }

const ui = {
  panel: '#0f0f11',
  border: '#22232a',
  text: '#eaeaea',
  textDim: '#9ca3af',
  chipBg: '#181a1f',
};

const shorten = (v?: string, len = 4) =>
  !v ? '' : v.length <= len * 2 + 3 ? v : `${v.slice(0, len)}...${v.slice(-len)}`;

const relTime = (t: number) => {
  const d = new Date(t * 1000), diff = Date.now() - d.getTime();
  if (diff < 60_000) return 'now';
  if (diff < 3_600_000) return `${Math.floor(diff/60_000)}m`;
  if (diff < 86_400_000) return `${Math.floor(diff/3_600_000)}h`;
  if (diff < 604_800_000) return `${Math.floor(diff/86_400_000)}d`;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }).toLowerCase();
};

const epoch = (m: any) =>
  typeof m?.updated_at === 'number' ? m.updated_at :
  typeof m?.updated_at === 'string' ? Math.floor(new Date(m.updated_at).getTime()/1000) :
  typeof m?.time === 'number' ? m.time : Math.floor(Date.now()/1000);

const ownerOf = (m: any) => m?.owner || m?.creator || m?.author || m?.admin || 'unknown';

const fnsOf = (m: any): string[] => {
  const si = m?.schema?.input;
  if (si && typeof si === 'object') return Object.keys(si);
  if (Array.isArray(m?.functions)) return m.functions;
  if (m?.functions && typeof m.functions === 'object') return Object.keys(m.functions);
  if (m?.schema && typeof m.schema === 'object') return Object.keys(m.schema);
  return [];
};

// gradient cover fallback
const hueFrom = (s: string) => { let h = 0; for (let i=0;i<s.length;i++) h=(h*31+s.charCodeAt(i))>>>0; return h%360; };
const coverDataUri = (a: string, b: string, w = 384, h = 192) => {
  const h1 = hueFrom(a), h2 = (hueFrom(b) + 40) % 360;
  const svg = encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="hsl(${h1},70%,48%)"/>
          <stop offset="100%" stop-color="hsl(${h2},65%,34%)"/>
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill="url(#g)"/>
    </svg>`
  );
  return `data:image/svg+xml,${svg}`;
};

const chip = 'inline-flex items-center gap-1 rounded border px-2 py-1 text-[11px] leading-none';
const chipSolidStyle = (bg: string, br: string, fg: string) => ({ backgroundColor: bg, borderColor: br, color: fg });
const chipMutedStyle = (br: string, fg: string) => ({ backgroundColor: 'transparent', borderColor: br, color: fg });

const ModuleCard = memo(({ module }: ModuleCardProps) => {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  const time = useMemo(() => epoch(module as any), [module]);
  const owner = useMemo(() => ownerOf(module as any), [module]);
  const funcs = useMemo(() => fnsOf(module as any), [module]);

  const onOpen = useCallback((e: React.MouseEvent | React.KeyboardEvent) => {
    e.stopPropagation(); setBusy(true); router.push(`${module.name}`);
  }, [router, module.name]);

  const coverUrl =
    (module as any)?.image ||
    (module as any)?.banner ||
    coverDataUri(owner || module.name || 'seed', module.name || owner || 'seed');

  // shared edge-fade for scroll rows (softens edges + scrollbar)
  const fadeMask: React.CSSProperties = {
    WebkitMaskImage: 'linear-gradient(to right, transparent, black 10px, black calc(100% - 10px), transparent)',
    maskImage: 'linear-gradient(to right, transparent, black 10px, black calc(100% - 10px), transparent)',
  };

  return (
    <div
      role="button" tabIndex={0}
      onClick={onOpen as any}
      onKeyDown={(e) => ((e.key === 'Enter' || e.key === ' ') ? onOpen(e) : null)}
      className="relative flex flex-col overflow-hidden rounded-xl border md:flex-row"
      style={{ backgroundColor: ui.panel, borderColor: ui.border }}
      aria-label={`Open ${module.name}`}
    >
      {busy && (
        <div className="absolute inset-0 z-20 m-0 flex items-center justify-center bg-black/60">
          <div className="animate-pulse text-[11px] font-bold text-white">loading…</div>
        </div>
      )}

      {/* LEFT: attributes / metadata */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* row 1: name | time */}
        <div className="flex items-center justify-between gap-2 px-3 pt-2">
          <h3 className="truncate text-[13px] font-semibold leading-none" style={{ color: ui.text }} title={module.name}>
            {module.name}
          </h3>
          <span className={chip} style={chipMutedStyle(ui.border, ui.textDim)} title={new Date(time*1000).toLocaleString()}>
            <Clock3 className="h-3.5 w-3.5" /> {relTime(time)}
          </span>
        </div>

        {/* row 2: key, cid, tags inline (subtle scrollbar) */}
        <div
          className="micro-scroll -mx-1 flex max-w-full items-center gap-1 overflow-x-auto px-1 pt-1"
          style={fadeMask}
        >
          <span className={chip} style={chipSolidStyle(ui.chipBg, ui.border, ui.text)} title={`key: ${module.key}`}>
            <TagIcon className="h-3.5 w-3.5 opacity-90" /><code>{shorten(module.key, 6)}</code>
          </span>
          <CopyButton size="xs" code={module.key} />
          {(module as any)?.cid && (
            <>
              <span className={chip} style={chipSolidStyle(ui.chipBg, ui.border, ui.text)} title={`cid: ${(module as any).cid}`}>
                <Boxes className="h-3.5 w-3.5 opacity-90" /><code>{shorten((module as any).cid, 6)}</code>
              </span>
              <CopyButton size="xs" code={(module as any).cid} />
            </>
          )}
          {module.tags?.length ? (
            <>
              {module.tags.slice(0, 12).map((t) => (
                <span key={t} className={`${chip} border-dashed`} style={chipMutedStyle(ui.border, ui.textDim)} title={t}>
                  #{t}
                </span>
              ))}
              {module.tags.length > 12 && (
                <span className={chip} style={chipMutedStyle(ui.border, ui.textDim)}>+{module.tags.length - 12}</span>
              )}
            </>
          ) : (
            <span className={`${chip} border-dashed`} style={chipMutedStyle(ui.border, ui.textDim)}>#untagged</span>
          )}
        </div>

        {/* row 3: desc | owner | functions */}
        <div
          className="micro-scroll -mx-1 flex max-w-full items-center gap-1 overflow-x-auto px-1 pb-3 pt-1"
          style={fadeMask}
        >
          {(module.description || (module as any).desc) && (
            <span className="truncate text-[11px] leading-none" style={{ color: ui.textDim }}>
              {module.description || (module as any).desc}
            </span>
          )}
          <span className={chip} style={chipSolidStyle(ui.chipBg, ui.border, ui.text)} title={`owner: ${owner}`}>
            <User className="h-3.5 w-3.5 opacity-90" /><code>{shorten(owner, 6)}</code>
          </span>
          <CopyButton size="xs" code={String(owner)} />

          {!!funcs.length && (
            <>
              <span className="text-[10.5px] leading-none" style={{ color: ui.textDim }}>functions</span>
              {funcs.slice(0, 14).map((fn) => (
                <span
                  key={fn}
                  className="rounded border px-2 py-1 text-[10.5px] leading-none"
                  style={{ borderColor: ui.border, backgroundColor: ui.chipBg, color: ui.text }}
                  title={fn}
                >
                  {fn}
                </span>
              ))}
              {funcs.length > 14 && (
                <span className="px-2 py-1 text-[10.5px]" style={{ color: ui.textDim }}>
                  +{funcs.length - 14}
                </span>
              )}
            </>
          )}
        </div>
      </div>

      {/* RIGHT: image */}
      <div className="relative w-full shrink-0 md:w-56">
        <div className="absolute inset-0">
          <img
            src={String(coverUrl)}
            alt=""
            className="h-full w-full object-cover"
            style={{ filter: 'saturate(1.05) contrast(1.02)' }}
            loading="lazy"
            decoding="async"
            referrerPolicy="no-referrer"
          />
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-l from-black/14 via-transparent to-transparent md:from-black/10" />
        </div>
        {/* spacer to set right column height; matches left’s natural height on mobile and fills on md+ */}
        <div className="invisible aspect-[16/9] w-full md:aspect-auto md:h-full" />
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
      `}</style>
    </div>
  );
});

ModuleCard.displayName = 'ModuleCard';
export default ModuleCard;
