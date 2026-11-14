"use client";

import { geoAlbersUsa, geoPath } from "d3-geo";
import { useMemo, useState } from "react";
import { feature, mesh } from "topojson-client";
// Pre-projected (Albers USA, 975x610) state outlines from the US Census, via us-atlas
import statesTopo from "us-atlas/states-albers-10m.json";

export type MapMarket = { city: string; label: string; lat: number; lon: number; listings: number; median_price: number };

// Same projection the atlas was built with, so market dots line up with the outlines
const projection = geoAlbersUsa().scale(1300).translate([487.5, 305]);
const VB = { x: -150, y: 20, w: 1250, h: 600 };
const RAMP = ["#C5D2EE", "#9DB2E0", "#6F8BC9", "#3D5A99", "#0F1F3D"];

const SHORT: Record<string, string> = {
  "Washington Dc, DC": "Washington, DC",
  "Clark County, NV": "Las Vegas",
  "Hawaii, HI": "Hawaii",
  "Rhode Island, RI": "Rhode Island",
  "Broward County, FL": "Broward County",
  "San Mateo County, CA": "San Mateo County",
  "Santa Clara County, CA": "Santa Clara County",
  "Santa Cruz County, CA": "Santa Cruz County",
  "New York City, NY": "New York City",
};
const shortName = (city: string) => SHORT[city] ?? city.replace(/, [A-Z]{2}$/, "");

// Label placement. Crowded markets get a callout over the ocean with a leader line.
type Place = { dx?: number; dy?: number; anchor?: "start" | "end" | "middle"; callout?: [number, number] };
const PLACE: Record<string, Place> = {
  "San Francisco, CA": { callout: [-12, 222], anchor: "end" },
  "Oakland, CA": { callout: [-12, 240], anchor: "end" },
  "San Mateo County, CA": { callout: [-12, 258], anchor: "end" },
  "Santa Clara County, CA": { callout: [-12, 276], anchor: "end" },
  "Santa Cruz County, CA": { callout: [-12, 294], anchor: "end" },
  "Pacific Grove, CA": { callout: [-12, 312], anchor: "end" },
  "Los Angeles, CA": { dx: -12, dy: 4, anchor: "end" },
  "San Diego, CA": { dx: -12, dy: 6, anchor: "end" },
  "Portland, OR": { dx: -10, dy: 4, anchor: "end" },
  "Seattle, WA": { dx: -10, dy: 4, anchor: "end" },
  "Cambridge, MA": { callout: [975, 140], anchor: "start" },
  "Boston, MA": { callout: [975, 158], anchor: "start" },
  "Rhode Island, RI": { callout: [975, 184], anchor: "start" },
  "New York City, NY": { callout: [975, 210], anchor: "start" },
  "Jersey City, NJ": { callout: [975, 228], anchor: "start" },
  "Washington Dc, DC": { callout: [975, 268], anchor: "start" },
  "Rochester, NY": { dx: -9, dy: -6, anchor: "end" },
  "Fort Worth, TX": { dx: -10, dy: -6, anchor: "end" },
  "Dallas, TX": { dx: 10, dy: -6, anchor: "start" },
  "Austin, TX": { dx: -10, dy: 4, anchor: "end" },
  "Broward County, FL": { dx: -10, dy: 4, anchor: "end" },
  "Columbus, OH": { dx: 9, dy: -6, anchor: "start" },
  "Asheville, NC": { dx: 9, dy: 10, anchor: "start" },
};

export default function UsMarketMap({
  markets, selected, onSelect,
}: { markets: MapMarket[]; selected: string; onSelect: (city: string) => void }) {
  const [hover, setHover] = useState<string | null>(null);

  const { statePaths, borders, nation } = useMemo(() => {
    const topo = statesTopo as any;
    const path = geoPath();
    const states = (feature(topo, topo.objects.states) as any).features as any[];
    return {
      statePaths: states.map((s) => path(s) as string),
      borders: path(mesh(topo, topo.objects.states, (a: any, b: any) => a !== b)) as string,
      nation: path(feature(topo, topo.objects.nation) as any) as string,
    };
  }, []);

  const pts = useMemo(() => {
    const prices = markets.map((m) => m.median_price).sort((a, b) => a - b);
    const q = (p: number) => prices[Math.min(prices.length - 1, Math.floor(p * prices.length))];
    const breaks = [q(0.2), q(0.4), q(0.6), q(0.8)];
    return markets
      .map((m) => {
        const xy = projection([m.lon, m.lat]);
        if (!xy) return null;
        const bin = breaks.filter((b) => m.median_price >= b).length;
        return { ...m, x: xy[0], y: xy[1], r: 3.5 + Math.sqrt(m.listings) / 70, color: RAMP[bin], place: PLACE[m.city] ?? { dx: 9, dy: 4, anchor: "start" as const } };
      })
      .filter(Boolean) as (MapMarket & { x: number; y: number; r: number; color: string; place: Place })[];
  }, [markets]);

  const hovered = pts.find((p) => p.city === hover);
  // Draw the selected and hovered markets last so they sit on top
  const ordered = [...pts].sort((a, b) => Number(a.city === selected || a.city === hover) - Number(b.city === selected || b.city === hover) || b.r - a.r);

  return (
    <div className="relative">
      <svg viewBox={`${VB.x} ${VB.y} ${VB.w} ${VB.h}`} className="h-auto w-full select-none" role="group" aria-label="Map of covered markets">
        <path d={nation} fill="#EEF1F5" />
        {statePaths.map((d, i) => <path key={i} d={d} fill="#EEF1F5" />)}
        <path d={borders} fill="none" stroke="#FFFFFF" strokeWidth={1.2} strokeLinejoin="round" />
        <path d={nation} fill="none" stroke="#D5DBE4" strokeWidth={0.8} />

        {ordered.map((p) => {
          const active = p.city === selected;
          const isHover = p.city === hover;
          const pl = p.place;
          const lx = pl.callout ? pl.callout[0] : p.x + (pl.dx ?? 9);
          const ly = pl.callout ? pl.callout[1] : p.y + (pl.dy ?? 4);
          const select = () => onSelect(p.city);
          return (
            <g key={p.city} className="cursor-pointer" role="button" tabIndex={0} aria-pressed={active}
              aria-label={`${p.label}: median $${Math.round(p.median_price)} per night, ${p.listings.toLocaleString()} listing snapshots`}
              onClick={select} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), select())}
              onMouseEnter={() => setHover(p.city)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(p.city)} onBlur={() => setHover(null)}>
              {pl.callout && (
                <path className="hidden sm:inline" d={`M${p.x},${p.y} L${lx + (pl.anchor === "end" ? 4 : -4)},${ly - 3}`} stroke={active || isHover ? "#0F1F3D" : "#B4BDCB"} strokeWidth={0.8} fill="none" />
              )}
              {active && <circle cx={p.x} cy={p.y} r={p.r + 6} fill="#E0322F" opacity={0.18} />}
              <circle cx={p.x} cy={p.y} r={active ? p.r + 1.5 : p.r} fill={active ? "#E0322F" : p.color} stroke="#fff" strokeWidth={1.5} opacity={0.95} />
              <circle cx={p.x} cy={p.y} r={Math.max(p.r + 4, 9)} fill="transparent" />
              <text x={lx} y={ly} textAnchor={pl.anchor ?? "start"} className="hidden sm:inline"
                style={{ font: `${active || isHover ? 700 : 500} 11px Inter Variable, Inter, system-ui, sans-serif`, fill: active ? "#E0322F" : isHover ? "#0F1F3D" : "#5B6577", paintOrder: "stroke", stroke: "#fff", strokeWidth: 3, strokeLinejoin: "round" }}>
                {shortName(p.city)}
              </text>
              {pl.callout && (
                <rect x={pl.anchor === "end" ? lx - 140 : lx - 2} y={ly - 12} width={142} height={16} fill="transparent" />
              )}
            </g>
          );
        })}
      </svg>

      {hovered && (
        <div className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-md border border-line bg-white px-3 py-2 text-xs shadow-card"
          style={{ left: `${((hovered.x - VB.x) / VB.w) * 100}%`, top: `calc(${((hovered.y - VB.y) / VB.h) * 100}% - 12px)` }}>
          <div className="font-semibold text-ink">{hovered.label}</div>
          <div className="tnum text-muted">Median ${Math.round(hovered.median_price)} a night · {hovered.listings.toLocaleString()} snapshots</div>
        </div>
      )}
    </div>
  );
}

export const MAP_RAMP = RAMP;
