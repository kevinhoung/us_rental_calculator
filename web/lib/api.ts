export type Coverage = {
  covered: boolean;
  market: string | null;
  label: string | null;
  state: string | null;
  nearest_km: number | null;
};

export type Place = {
  label: string;
  lat: number;
  lon: number;
  state?: string | null;
  coverage: Coverage;
};

export type Market = {
  city: string;
  label: string;
  state: string;
  lat: number;
  lon: number;
  listings: number;
  median_price: number;
};

export type Metrics = {
  model: string;
  features: string[];
  evaluated_rows: number;
  median_abs_error_usd: number;
  mean_abs_error_usd: number;
  median_pct_error: number;
  share_within_25pct: number;
  baseline_city_median_pct_error: number;
  caveat: string;
  computed_on: string;
};

export type Meta = {
  markets: Market[];
  categories: string[];
  listings: number;
  metrics: Metrics;
  data_source: string;
};

export type EstimateInput = {
  lat: number;
  lon: number;
  category: string;
  guests: number;
  bedrooms: number;
  beds: number;
  bathrooms: number;
  cleanliness: number;
  location_score: number;
  instant_book: boolean;
};

export type Estimate = {
  price: number;
  coverage: Coverage;
  comps: {
    radius_km: number;
    count: number;
    p25: number | null;
    median: number | null;
    p75: number | null;
    similar_count: number;
    similar_p25: number | null;
    similar_median: number | null;
    similar_p75: number | null;
    percentile_of_estimate: number | null;
  };
  points: { lat: number; lon: number; price: number; cat: string; bedrooms: number | null }[];
  tax_rate_pct: number;
  model_error_pct: number;
};

export type Rates = { rate30: number | null; rate15: number | null; as_of: string | null; source: string | null; live: boolean };

export class ApiError extends Error {}

// When set (production), the browser calls the API host directly. Otherwise /api is proxied by next.config rewrites.
const BASE = (process.env.NEXT_PUBLIC_API_BASE || "").replace(/\/$/, "");

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, {
      ...init,
      headers: init?.body ? { "content-type": "application/json", ...(init?.headers || {}) } : init?.headers,
    });
  } catch {
    throw new ApiError("The pricing service is unreachable right now. Try again in a minute.");
  }
  if (!res.ok) {
    let msg = "Something went wrong. Try again in a minute.";
    try {
      const j = await res.json();
      if (typeof j?.detail === "string") msg = j.detail;
    } catch {}
    throw new ApiError(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => req<Meta>("/api/meta"),
  rates: () => req<Rates>("/api/rates"),
  suggest: (q: string, signal?: AbortSignal) => req<{ items: Place[] }>(`/api/suggest?q=${encodeURIComponent(q)}`, { signal }),
  geocode: (q: string) => req<Place>(`/api/geocode?q=${encodeURIComponent(q)}`),
  estimate: (body: EstimateInput) => req<Estimate>("/api/estimate", { method: "POST", body: JSON.stringify(body) }),
  market: (p: { city: string; cat: string; superhost: string; quarter: string }) =>
    req<any>(`/api/market?${new URLSearchParams(p).toString()}`),
  marketFilters: () => req<{ markets: { city: string; label: string }[]; categories: string[]; quarters: number[] }>("/api/market/filters"),
};

export const usd = (n: number | null | undefined, digits = 0) =>
  n == null || Number.isNaN(n)
    ? "n/a"
    : n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: digits, minimumFractionDigits: digits });

export const signedUsd = (n: number) => (n >= 0 ? "+" : "−") + usd(Math.abs(n));
export const pct = (n: number | null | undefined, digits = 0) => (n == null ? "n/a" : `${n.toFixed(digits)}%`);
