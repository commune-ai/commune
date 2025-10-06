'use client';

import { memo, useCallback, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { CopyButton } from '@/app/components/CopyButton';
import { ModuleType } from '@app/types/module';
import { User, Clock3, Boxes } from 'lucide-react';

interface ModuleCardProps { module: ModuleType }

const ui = {
  panel: '#0f0f11',
  border: '#22232a',
  text: '#eaeaea',
  textDim: '#9ca3af',
  chipBg: '#181a1f',
};

const shorten = (v?: string, len = 6) =>
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

const chip = 'inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-base font-medium leading-tight';
const chipStyle = (bg: string, br: string, fg: string) => ({ backgroundColor: bg, borderColor: br, color: fg });

const ModuleCard = memo(({ module }: ModuleCardProps) => {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const time = useMemo(() => epoch(module as any), [module]);
  const owner = useMemo(() => ownerOf(module as any), [module]);
  const cid = useMemo(() => (module as any)?.cid || null, [module]);
  const description = useMemo(() => (module as any)?.description || (module as any)?.desc || 'No description available', [module]);

  const onOpen = useCallback((e: React.MouseEvent | React.KeyboardEvent) => {
    e.stopPropagation(); setBusy(true); router.push(`${module.name}`);
  }, [router, module.name]);

  const coverUrl =
    (module as any)?.image ||
    (module as any)?.banner ||
    coverDataUri(owner || module.name || 'seed', module.name || owner || 'seed');

  return (
    <div
      role="button" tabIndex={0}
      onClick={onOpen as any}
      onKeyDown={(e) => ((e.key === 'Enter' || e.key === ' ') ? onOpen(e) : null)}
      className="group relative flex flex-col overflow-hidden rounded-xl border h-full transition-all hover:shadow-2xl hover:scale-[1.02]"
      style={{ backgroundColor: ui.panel, borderColor: ui.border }}
      aria-label={`Open ${module.name}`}
    >
      {busy && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="animate-pulse text-lg font-bold text-white">loading…</div>
        </div>
      )}

      <div className="relative w-full h-48 overflow-hidden">
        <img
          src={String(coverUrl)}
          alt=""
          className="h-full w-full object-cover transition-transform group-hover:scale-110"
          style={{ filter: 'saturate(1.1) contrast(1.05)' }}
          loading="lazy"
          decoding="async"
          referrerPolicy="no-referrer"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
        
        <div className="absolute bottom-3 left-3 right-3">
          <h3 className="text-2xl font-bold leading-tight mb-2" style={{ color: ui.text }} title={module.name}>
            {module.name}
          </h3>
        </div>
      </div>

      <div className="flex gap-3 p-5">
        <div className="flex flex-col gap-3 flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className={chip} style={chipStyle(ui.chipBg, ui.border, ui.text)} title={`Owner: ${owner}`}>
              <User className="h-5 w-5" />
              <code className="text-sm">{shorten(owner, 5)}</code>
            </span>
            <CopyButton size="xs" code={String(owner)} />
          </div>

          {cid && (
            <div className="flex items-center gap-2">
              <span className={chip} style={chipStyle(ui.chipBg, ui.border, ui.text)} title={`CID: ${cid}`}>
                <Boxes className="h-5 w-5" />
                <code className="text-sm">{shorten(cid, 5)}</code>
              </span>
              <CopyButton size="xs" code={cid} />
            </div>
          )}

          <div className="flex items-center gap-2">
            <span className={chip} style={chipStyle(ui.chipBg, ui.border, ui.textDim)} title={new Date(time*1000).toLocaleString()}>
              <Clock3 className="h-5 w-5" />
              <span className="text-sm">{relTime(time)}</span>
            </span>
          </div>
        </div>

        <div className="flex-1 min-w-0 pt-2 border-l pl-3" style={{ borderColor: ui.border }}>
          <div className="relative">
            <p 
              className={`text-sm leading-relaxed transition-all duration-300 ${expanded ? '' : 'line-clamp-3'}`}
              style={{ color: ui.textDim }}
            >
              {description}
            </p>
            {description.length > 100 && (
              <button
                onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}
                className="mt-1 text-sm font-medium transition-colors hover:underline"
                style={{ color: ui.text }}
              >
                {expanded ? 'Show less' : 'Read more'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
});

ModuleCard.displayName = 'ModuleCard';
export default ModuleCard;