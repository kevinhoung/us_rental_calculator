"use client";

import { Check, MapPin, Search, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, Market, Place } from "@/lib/api";

export function AddressSearch({
  value, onSelect, markets,
}: { value: Place | null; onSelect: (p: Place | null) => void; markets: Market[] }) {
  const [q, setQ] = useState(value?.label ?? "");
  const [items, setItems] = useState<Place[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showMarkets, setShowMarkets] = useState(false);
  const ctrl = useRef<AbortController | null>(null);

  useEffect(() => {
    if (value && value.label !== q) setQ(value.label);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  useEffect(() => {
    if (value && q === value.label) return;
    if (q.trim().length < 3) {
      setItems([]);
      return;
    }
    const t = setTimeout(async () => {
      ctrl.current?.abort();
      ctrl.current = new AbortController();
      try {
        const r = await api.suggest(q, ctrl.current.signal);
        setItems(r.items);
        setOpen(true);
        setActive(-1);
      } catch {
        /* suggestions are optional; Enter still geocodes */
      }
    }, 280);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q]);

  async function resolveTyped() {
    if (q.trim().length < 2) return;
    setBusy(true);
    setErr(null);
    try {
      const p = await api.geocode(q);
      onSelect(p);
      setQ(p.label);
      setOpen(false);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  function pick(p: Place) {
    onSelect(p);
    setQ(p.label);
    setOpen(false);
    setErr(null);
  }

  return (
    <div className="relative">
      <label htmlFor="addr" className="mb-1.5 block text-[13px] font-medium text-text">Address or city</label>
      <div className="flex h-11 items-center gap-2 rounded-sm border border-line bg-white px-3 focus-within:border-primary focus-within:ring-2 focus-within:ring-primary-soft">
        <Search size={16} className="shrink-0 text-faint" />
        <input
          id="addr" autoComplete="off" role="combobox" aria-expanded={open} aria-controls="addr-list"
          className="w-full bg-transparent text-[15px] outline-none placeholder:text-faint"
          placeholder="1420 Larimer St, Denver, CO" value={q}
          onChange={(e) => { setQ(e.target.value); if (value) onSelect(null); }}
          onFocus={() => items.length && setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(items.length - 1, a + 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(-1, a - 1)); }
            if (e.key === "Enter") { e.preventDefault(); if (open && active >= 0) pick(items[active]); else resolveTyped(); }
            if (e.key === "Escape") setOpen(false);
          }}
        />
        {busy && <span className="h-4 w-4 animate-spin rounded-full border-2 border-line border-t-primary" />}
      </div>

      {open && items.length > 0 && (
        <ul id="addr-list" role="listbox" className="absolute z-20 mt-1 w-full overflow-hidden rounded-md border border-line bg-white shadow-card">
          {items.map((it, i) => (
            <li key={`${it.label}-${i}`} role="option" aria-selected={i === active}>
              <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => pick(it)}
                className={`flex w-full items-start gap-2 px-3 py-2.5 text-left text-sm ${i === active ? "bg-sunken" : "hover:bg-sunken"}`}>
                <MapPin size={15} className="mt-0.5 shrink-0 text-faint" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-text">{it.label}</span>
                  <span className={`text-xs ${it.coverage.covered ? "text-pos" : "text-faint"}`}>
                    {it.coverage.covered ? `In ${it.coverage.label}` : "Outside the 30 covered markets"}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-2 min-h-[18px] text-xs">
        {err ? (
          <span className="flex items-center gap-1.5 text-neg"><TriangleAlert size={13} /> {err}</span>
        ) : value ? (
          value.coverage.covered ? (
            <span className="flex items-center gap-1.5 text-pos"><Check size={13} /> {value.coverage.label} is one of the 30 covered markets</span>
          ) : (
            <span className="flex items-center gap-1.5 text-warn"><TriangleAlert size={13} /> Outside the covered markets, so the estimate is less reliable</span>
          )
        ) : (
          <button type="button" className="text-primary hover:underline" onClick={() => setShowMarkets((s) => !s)}>
            {showMarkets ? "Hide covered markets" : "Or pick one of the 30 covered markets"}
          </button>
        )}
      </div>

      {showMarkets && !value && (
        <select
          className="mt-2 h-10 w-full rounded-sm border border-line bg-white px-2 text-sm"
          defaultValue=""
          onChange={(e) => {
            const m = markets.find((x) => x.city === e.target.value);
            if (m) {
              pick({ label: m.label, lat: m.lat, lon: m.lon, state: m.state,
                coverage: { covered: true, market: m.city, label: m.label, state: m.state, nearest_km: 0 } });
              setShowMarkets(false);
            }
          }}
          aria-label="Covered markets"
        >
          <option value="" disabled>Choose a market (uses its center)</option>
          {markets.map((m) => <option key={m.city} value={m.city}>{m.label}</option>)}
        </select>
      )}
    </div>
  );
}
