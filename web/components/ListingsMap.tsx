"use client";

import { useEffect, useRef, useState } from "react";

const STYLE = "https://tiles.openfreemap.org/styles/positron";
const RAMP = ["#C5D2EE", "#9DB2E0", "#6F8BC9", "#3D5A99", "#0F1F3D"];

export type MapPoint = { lat: number; lon: number; price: number };

function quantiles(prices: number[]) {
  const s = [...prices].sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))] ?? 0;
  return [q(0.2), q(0.4), q(0.6), q(0.8)];
}

export default function ListingsMap({
  points, center, zoom = 12.5, height = 300, showCenter = true,
}: { points: MapPoint[]; center: { lat: number; lon: number }; zoom?: number; height?: number; showCenter?: boolean }) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<any>(null);
  const [failed, setFailed] = useState(false);
  const breaks = quantiles(points.map((p) => p.price));

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const maplibregl = (await import("maplibre-gl")).default;
        if (cancelled || !el.current) return;
        const m = new maplibregl.Map({
          container: el.current, style: STYLE, center: [center.lon, center.lat], zoom,
          attributionControl: { compact: true }, cooperativeGestures: true,
        });
        map.current = m;
        m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
        m.on("error", (e: any) => { if (String(e?.error?.message || "").includes("style")) setFailed(true); });
        m.on("load", () => {
          m.addSource("pts", { type: "geojson", data: toGeo(points) });
          m.addLayer({
            id: "pts", type: "circle", source: "pts",
            paint: {
              "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 2.5, 14, 5],
              "circle-color": ["step", ["get", "price"], RAMP[0], breaks[0], RAMP[1], breaks[1], RAMP[2], breaks[2], RAMP[3], breaks[3], RAMP[4]],
              "circle-opacity": 0.85, "circle-stroke-color": "#fff", "circle-stroke-width": 0.6,
            },
          });
          const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });
          m.on("mouseenter", "pts", (e: any) => {
            m.getCanvas().style.cursor = "pointer";
            const f = e.features[0];
            popup.setLngLat(f.geometry.coordinates).setHTML(`<b>$${f.properties.price}</b> per night`).addTo(m);
          });
          m.on("mouseleave", "pts", () => { m.getCanvas().style.cursor = ""; popup.remove(); });
          if (showCenter) {
            const dot = document.createElement("div");
            dot.style.cssText = "width:16px;height:16px;border-radius:999px;background:#E0322F;border:3px solid #fff;box-shadow:0 0 0 6px rgba(224,50,47,.2)";
            new maplibregl.Marker({ element: dot }).setLngLat([center.lon, center.lat]).addTo(m);
          }
        });
      } catch {
        setFailed(true);
      }
    })();
    return () => { cancelled = true; map.current?.remove(); map.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [center.lat, center.lon, points]);

  return (
    <div>
      <div className="relative overflow-hidden rounded-md border border-line bg-sunken" style={{ height }}>
        <div ref={el} className="absolute inset-0" />
        {failed && (
          <div className="absolute inset-0 grid place-items-center p-6 text-center text-sm text-muted">
            The map is unavailable right now. The numbers on this page are not affected.
          </div>
        )}
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
        {showCenter ? (
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-brand" /> Your property</span>
        ) : <span />}
        <span className="flex items-center gap-1.5">
          <span className="flex">{RAMP.map((c) => <span key={c} className="h-2.5 w-4" style={{ background: c }} />)}</span>
          <span className="tnum">${Math.round(breaks[0] || 0)} to ${Math.round(breaks[3] || 0)}+ per night</span>
        </span>
      </div>
    </div>
  );
}

function toGeo(points: MapPoint[]) {
  return {
    type: "FeatureCollection" as const,
    features: points.map((p) => ({ type: "Feature" as const, properties: { price: p.price }, geometry: { type: "Point" as const, coordinates: [p.lon, p.lat] } })),
  };
}
