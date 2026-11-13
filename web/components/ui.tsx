"use client";

import { Minus, Plus } from "lucide-react";

export function Stepper({
  label, value, onChange, min, max, step = 1, format,
}: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; step?: number; format?: (v: number) => string;
}) {
  const set = (v: number) => onChange(Math.min(max, Math.max(min, Math.round(v / step) * step)));
  return (
    <div>
      <div className="mb-1.5 text-[13px] font-medium text-text">{label}</div>
      <div className="flex h-10 items-center rounded-sm border border-line bg-white">
        <span className="tnum flex-1 px-3 text-[15px]" aria-live="polite">{format ? format(value) : value}</span>
        <button type="button" aria-label={`Decrease ${label}`} onClick={() => set(value - step)} disabled={value <= min}
          className="grid h-10 w-10 place-items-center border-l border-line text-muted hover:bg-sunken disabled:opacity-40">
          <Minus size={14} />
        </button>
        <button type="button" aria-label={`Increase ${label}`} onClick={() => set(value + step)} disabled={value >= max}
          className="grid h-10 w-10 place-items-center border-l border-line text-muted hover:bg-sunken disabled:opacity-40">
          <Plus size={14} />
        </button>
      </div>
    </div>
  );
}

export function Slider({
  label, value, onChange, min, max, step = 1, display, hint,
}: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; step?: number; display: string; hint?: React.ReactNode;
}) {
  const fill = ((value - min) / (max - min)) * 100;
  return (
    <label className="block">
      <div className="mb-2 flex items-baseline justify-between gap-2 text-[13px]">
        <span className="font-medium text-text">
          {label} {hint && <span className="font-normal text-faint">{hint}</span>}
        </span>
        <span className="tnum font-semibold text-ink">{display}</span>
      </div>
      <input type="range" className="slider" min={min} max={max} step={step} value={value}
        style={{ ["--fill" as any]: `${fill}%` }} onChange={(e) => onChange(Number(e.target.value))} aria-valuetext={display} />
    </label>
  );
}

export function Segmented<T extends string>({
  options, value, onChange, cols = 2,
}: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void; cols?: number }) {
  return (
    <div className={`grid gap-2`} style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }} role="radiogroup">
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value} onClick={() => onChange(o.value)}
          className={`h-10 rounded-sm border px-2 text-[13px] font-medium transition-colors ${
            value === o.value ? "border-primary bg-primary-soft text-primary" : "border-line bg-white text-muted hover:border-faint"
          }`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function NumberField({
  label, value, onChange, prefix, suffix, step = 1, min = 0,
}: { label: string; value: number; onChange: (v: number) => void; prefix?: string; suffix?: string; step?: number; min?: number }) {
  return (
    <label className="block">
      <div className="mb-1.5 text-[13px] font-medium text-text">{label}</div>
      <div className="flex h-10 items-center rounded-sm border border-line bg-white px-3 focus-within:border-primary">
        {prefix && <span className="mr-1 text-muted">{prefix}</span>}
        <input type="number" inputMode="decimal" className="tnum w-full bg-transparent text-[15px] outline-none" value={Number.isFinite(value) ? value : ""}
          step={step} min={min} onChange={(e) => onChange(e.target.value === "" ? 0 : Number(e.target.value))} />
        {suffix && <span className="ml-1 text-muted">{suffix}</span>}
      </div>
    </label>
  );
}

export function Badge({ tone, children }: { tone: "pos" | "warn" | "neg" | "info"; children: React.ReactNode }) {
  const cls = {
    pos: "bg-pos-soft text-pos", warn: "bg-warn-soft text-warn", neg: "bg-neg-soft text-neg", info: "bg-primary-soft text-primary",
  }[tone];
  return <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${cls}`}>{children}</span>;
}

export function Chip({ children }: { children: React.ReactNode }) {
  return <span className="inline-flex items-center rounded-full border border-line bg-white px-3 py-1 text-xs font-medium text-muted">{children}</span>;
}

export function CardHeader({ title, sub, right }: { title: string; sub?: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="mb-4 flex items-start justify-between gap-3">
      <div>
        <h3 className="font-head text-[15px] font-bold text-ink">{title}</h3>
        {sub && <p className="mt-0.5 text-[13px] text-muted">{sub}</p>}
      </div>
      {right && <div className="shrink-0 text-xs text-faint">{right}</div>}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-md bg-sunken ${className}`} />;
}
