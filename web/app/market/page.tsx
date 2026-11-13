"use client";

import dynamic from "next/dynamic";
import { TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, Cell, LabelList, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Footer, Nav } from "@/components/Chrome";
import { CardHeader, Chip, Skeleton } from "@/components/ui";
import { api, usd } from "@/lib/api";

const ListingsMap = dynamic(() => import("@/components/ListingsMap"), { ssr: false, loading: () => <Skeleton className="h-[420px]" /> });

const TABS = ["Pricing", "Hosts", "Reviews", "Property mix", "Location"] as const;
type Tab = (typeof TABS)[number];
const INK = "#0F1F3D", BLUE = "#2456E6", SOFT = "#9DB2E0";
const tooltipStyle = { borderRadius: 8, border: "1px solid #E2E6EC", fontSize: 12 };

export default function MarketPage() {
  const [filters, setFilters] = useState({ city: "Denver, CO", cat: "all", superhost: "any", quarter: "all" });
  const [opts, setOpts] = useState<{ markets: { city: string; label: string }[]; categories: string[]; quarters: number[] } | null>(null);
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("Pricing");

  useEffect(() => { api.marketFilters().then(setOpts).catch(() => setError("Market data is unavailable right now. Try again in a minute.")); }, []);
  useEffect(() => {
    setLoading(true);
    api.market(filters).then((d) => { setData(d); setError(null); })
      .catch(() => setError("Market data is unavailable right now. Try again in a minute."))
      .finally(() => setLoading(false));
  }, [filters]);

  const label = opts?.markets.find((m) => m.city === filters.city)?.label ?? (filters.city === "all" ? "All 30 markets" : filters.city);
  const k = data?.kpis;
  const center = useMemo(() => {
    if (!data?.points?.length) return null;
    const lats = data.points.map((p: number[]) => p[0]).sort((a: number, b: number) => a - b);
    const lons = data.points.map((p: number[]) => p[1]).sort((a: number, b: number) => a - b);
    return { lat: lats[Math.floor(lats.length / 2)], lon: lons[Math.floor(lons.length / 2)] };
  }, [data]);

  const set = (key: keyof typeof filters) => (e: React.ChangeEvent<HTMLSelectElement>) => setFilters((f) => ({ ...f, [key]: e.target.value }));

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-[1200px] px-4 pt-6 sm:px-6 sm:pt-8">
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="font-head text-[24px] font-extrabold text-ink sm:text-[28px]">Market Insights</h1>
            <p className="mt-1 text-[15px] text-muted">Nightly prices, hosts and reviews across the 30 covered markets.</p>
          </div>
          <div className="flex flex-wrap gap-2"><Chip>883,120 listing snapshots</Chip><Chip>4 quarterly scrapes</Chip></div>
        </div>

        <div className="mt-5 flex flex-wrap gap-2">
          <Filter label="Market" value={filters.city} onChange={set("city")}
            options={[{ v: "all", l: "All markets" }, ...(opts?.markets.map((m) => ({ v: m.city, l: m.label })) ?? [{ v: filters.city, l: filters.city }])]} />
          <Filter label="Property type" value={filters.cat} onChange={set("cat")}
            options={[{ v: "all", l: "All" }, ...(opts?.categories.map((c) => ({ v: c, l: c })) ?? [])]} />
          <Filter label="Superhost" value={filters.superhost} onChange={set("superhost")}
            options={[{ v: "any", l: "All" }, { v: "yes", l: "Superhosts" }, { v: "no", l: "Other hosts" }]} />
          <Filter label="Quarter" value={filters.quarter} onChange={set("quarter")}
            options={[{ v: "all", l: "Q1 to Q4" }, ...(opts?.quarters.map((q) => ({ v: String(q), l: `Q${q}` })) ?? [])]} />
        </div>

        {error && (
          <div className="mt-5 flex items-center gap-2 rounded-md border border-line bg-white px-4 py-3 text-sm text-muted">
            <TriangleAlert size={16} className="text-warn" /> {error}
          </div>
        )}

        {data?.empty && <p className="mt-6 text-sm text-muted">No listings match these filters.</p>}

        <div className={`mt-5 grid grid-cols-2 gap-3 lg:grid-cols-4 lg:gap-4 ${loading ? "opacity-60" : ""}`}>
          {k ? (
            <>
              <Kpi label="Median nightly price" value={usd(k.median)} note={`middle 50%: ${usd(k.p25)} to ${usd(k.p75)}`} />
              <Kpi label="Listing snapshots" value={k.count.toLocaleString()} note="after filters" />
              <Kpi label="Superhost share" value={`${Math.round((k.superhost_share ?? 0) * 100)}%`}
                note={k.superhost_premium != null ? `${k.superhost_premium >= 0 ? "+" : "−"}${usd(Math.abs(k.superhost_premium))} median vs other hosts` : ""}
                tone={k.superhost_premium > 0 ? "text-pos" : undefined} />
              <Kpi label="Average rating" value={k.avg_rating?.toFixed(2) ?? "n/a"} note="out of 5" />
            </>
          ) : (
            Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-[104px]" />)
          )}
        </div>

        <div className="mt-6 flex gap-1 overflow-x-auto border-b border-line" role="tablist">
          {TABS.map((t) => (
            <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
              className={`whitespace-nowrap border-b-2 px-3 py-2.5 text-sm font-medium ${tab === t ? "border-primary text-primary" : "border-transparent text-muted hover:text-text"}`}>
              {t}
            </button>
          ))}
        </div>

        {data && !data.empty && (
          <div className={`mt-5 grid gap-4 lg:grid-cols-2 lg:items-start ${loading ? "opacity-60" : ""}`}>
            {tab === "Pricing" && (
              <>
                <div className="card p-5">
                  <CardHeader title="Median nightly price by market" sub="Same filters, every market. The selected market is the dark bar." />
                  <div style={{ height: Math.max(260, data.by_city.length * 22) }}>
                    <ResponsiveContainer>
                      <BarChart data={data.by_city} layout="vertical" margin={{ left: 0, right: 44, top: 0, bottom: 0 }}>
                        <XAxis type="number" hide />
                        <YAxis type="category" dataKey="label" width={196} tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: "#5B6577" }} interval={0} />
                        <Tooltip formatter={(v: any) => [usd(Number(v)), "Median"]} contentStyle={tooltipStyle} cursor={{ fill: "rgba(15,31,61,0.04)" }} />
                        <Bar dataKey="median" radius={[0, 4, 4, 0]} barSize={13}>
                          {data.by_city.map((d: any) => <Cell key={d.city} fill={d.city === filters.city ? INK : SOFT} />)}
                          <LabelList dataKey="median" position="right" formatter={(v: any) => usd(Number(v))} style={{ fontSize: 10, fill: "#5B6577" }} />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="card p-5">
                  <CardHeader title={`Price distribution, ${label}`}
                    sub={`Most listings fall between ${usd(k.p25)} and ${usd(k.p75)}. ${data.over_600.toLocaleString()} are $600 or more.`} />
                  <div className="h-[300px]">
                    <ResponsiveContainer>
                      <BarChart data={data.histogram} margin={{ top: 20, right: 4, left: 0, bottom: 0 }} barCategoryGap={1}>
                        <XAxis dataKey="from" tickFormatter={(v) => `$${v}`} tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: "#8A94A6" }} interval={3} />
                        <YAxis hide />
                        <Tooltip formatter={(v: any) => [Number(v).toLocaleString(), "Listings"]} labelFormatter={(l) => `$${l} to $${Number(l) + 25}`} contentStyle={tooltipStyle} cursor={{ fill: "rgba(15,31,61,0.04)" }} />
                        <ReferenceLine x={Math.floor(k.median / 25) * 25} stroke={INK} strokeDasharray="3 3" label={{ value: `Median ${usd(k.median)}`, position: "top", fontSize: 11, fill: INK }} />
                        <Bar dataKey="count" fill={SOFT} radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
            )}

            {tab === "Hosts" && (
              <>
                <div className="card p-5">
                  <CardHeader title="Superhosts compared with other hosts" sub={label} />
                  <table className="w-full text-sm">
                    <thead><tr className="text-left text-xs text-faint"><th className="pb-2 font-medium" /><th className="pb-2 text-right font-medium">Superhosts</th><th className="pb-2 text-right font-medium">Other hosts</th></tr></thead>
                    <tbody className="tnum">
                      {[
                        ["Listing snapshots", (h: any) => h.count.toLocaleString()],
                        ["Median nightly price", (h: any) => usd(h.median_price)],
                        ["Average rating", (h: any) => h.avg_rating?.toFixed(2) ?? "n/a"],
                        ["Instant book on", (h: any) => h.instant_book != null ? `${Math.round(h.instant_book * 100)}%` : "n/a"],
                        ["Median reviews a month", (h: any) => h.reviews_per_month?.toFixed(2) ?? "n/a"],
                      ].map(([name, f]: any) => (
                        <tr key={name} className="border-t border-line">
                          <td className="py-2.5 text-muted">{name}</td>
                          <td className="py-2.5 text-right font-semibold text-ink">{f(data.hosts[0])}</td>
                          <td className="py-2.5 text-right text-text">{f(data.hosts[1])}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="card p-5">
                  <CardHeader title="Median price by quarter" sub="Seasonality across the four scrapes." />
                  <SimpleBars data={data.by_quarter.map((d: any) => ({ name: `Q${d.quarter}`, value: d.median }))} />
                </div>
              </>
            )}

            {tab === "Reviews" && (
              <>
                <div className="card p-5">
                  <CardHeader title="Median price by overall rating" sub="Higher-rated listings charge more, but the gap is modest." />
                  <SimpleBars data={data.by_rating.map((d: any) => ({ name: d.band, value: d.median }))} />
                </div>
                <div className="card p-5">
                  <CardHeader title="Listings by rating band" sub="Most listings rate 4.8 or higher, so small differences matter." />
                  <SimpleBars data={data.by_rating.map((d: any) => ({ name: d.band, value: d.count }))} money={false} />
                </div>
              </>
            )}

            {tab === "Property mix" && (
              <>
                <div className="card p-5">
                  <CardHeader title="Median price by property type" sub={label} />
                  <SimpleBars data={data.by_category.map((d: any) => ({ name: d.cat.replace("Entire ", ""), value: d.median }))} />
                </div>
                <div className="card p-5">
                  <CardHeader title="Median price by bedrooms" sub="5 includes five or more." />
                  <SimpleBars data={data.by_bedrooms.map((d: any) => ({ name: `${d.bedrooms} bd`, value: d.median }))} />
                </div>
              </>
            )}

            {tab === "Location" && (
              <div className="card p-5 lg:col-span-2">
                <CardHeader title={`Where listings are, ${label}`} sub="Latest quarter, up to 1,500 listings, colored by nightly price." />
                {filters.city === "all" || !center ? (
                  <p className="text-sm text-muted">Pick a single market to see the map.</p>
                ) : (
                  <ListingsMap key={filters.city} points={data.points.map((p: number[]) => ({ lat: p[0], lon: p[1], price: p[2] }))}
                    center={center} zoom={10.5} height={420} showCenter={false} />
                )}
              </div>
            )}
          </div>
        )}
      </main>
      <Footer />
    </>
  );
}

function Filter({ label, value, onChange, options }: { label: string; value: string; onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void; options: { v: string; l: string }[] }) {
  return (
    <label className="flex h-10 items-center gap-2 rounded-sm border border-line bg-white pl-3 pr-1 text-[13px]">
      <span className="text-muted">{label}</span>
      <select value={value} onChange={onChange} className="h-full max-w-[200px] bg-transparent pr-1 font-semibold text-ink outline-none">
        {options.map((o) => <option key={o.v} value={o.v}>{o.l}</option>)}
      </select>
    </label>
  );
}

function Kpi({ label, value, note, tone }: { label: string; value: string; note: string; tone?: string }) {
  return (
    <div className="rounded-md border border-line bg-white p-4 shadow-card">
      <div className="text-[13px] text-muted">{label}</div>
      <div className="tnum mt-1 font-head text-[22px] font-bold text-ink sm:text-[24px]">{value}</div>
      <div className={`mt-0.5 text-xs ${tone ?? "text-faint"}`}>{note}</div>
    </div>
  );
}

function SimpleBars({ data, money = true }: { data: { name: string; value: number }[]; money?: boolean }) {
  const fmt = (v: number) => (money ? usd(v) : v.toLocaleString());
  return (
    <div className="h-[260px]">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 22, right: 4, left: 0, bottom: 0 }}>
          <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: "#5B6577" }} interval={0} />
          <YAxis hide />
          <Tooltip formatter={(v: any) => [fmt(Number(v)), money ? "Median" : "Listings"]} contentStyle={tooltipStyle} cursor={{ fill: "rgba(15,31,61,0.04)" }} />
          <Bar dataKey="value" fill={BLUE} radius={[4, 4, 0, 0]} maxBarSize={56}>
            <LabelList dataKey="value" position="top" formatter={(v: any) => fmt(Number(v))} style={{ fontSize: 11, fill: "#5B6577" }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
