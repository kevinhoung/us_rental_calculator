"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function LogoMark({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="#FCE9E7" />
      <path d="M8 15.5 16 9l8 6.5V24a1 1 0 0 1-1 1h-4.5v-5.5h-5V25H9a1 1 0 0 1-1-1v-8.5Z" fill="none" stroke="#E0322F" strokeWidth="2.2" strokeLinejoin="round" />
    </svg>
  );
}

const LINKS = [
  { href: "/", label: "Estimate" },
  { href: "/market", label: "Market Insights" },
  { href: "/method", label: "Method" },
  { href: "/about", label: "About" },
];

export function Nav({ actions }: { actions?: React.ReactNode }) {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-white/90 backdrop-blur print:hidden">
      <div className="mx-auto flex h-[60px] max-w-[1200px] items-center gap-3 px-4 sm:px-6">
        <Link href="/" className="flex shrink-0 items-center gap-2 font-head text-[17px] font-extrabold text-ink">
          <LogoMark />
          <span>Pricision AI</span>
        </Link>
        <nav className="ml-2 flex min-w-0 items-center gap-1 overflow-x-auto sm:ml-6" aria-label="Main">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path?.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`whitespace-nowrap rounded-md px-2.5 py-1.5 text-[13px] font-medium sm:px-3 sm:text-sm ${
                  active ? "bg-primary-soft text-primary" : "text-muted hover:bg-sunken hover:text-text"
                } ${l.href === "/method" || l.href === "/about" ? "hidden sm:inline-block" : ""}`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto hidden items-center gap-2 md:flex">{actions}</div>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="mx-auto mt-12 max-w-[1200px] px-4 pb-28 sm:px-6 md:pb-12 print:hidden">
      <div className="flex flex-col gap-6 border-t border-line pt-6 text-[13px] text-muted md:flex-row md:items-center md:justify-between">
        <p className="max-w-xl">
          <span className="font-semibold text-text">How this works.</span> A model trained on Airbnb listing snapshots
          across 30 US markets. Typical error and data sources are on the{" "}
          <Link href="/method" className="text-primary hover:underline">
            Method page
          </Link>
          . Estimates, not financial advice.
        </p>
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-full bg-primary-soft font-head text-xs font-bold text-primary">KH</div>
          <div>
            <div className="font-semibold text-text">Built by Kevin Houng</div>
            <div>
              Attorney turned HR tech lead and investor.{" "}
              <Link href="/about" className="text-primary hover:underline">
                Read the story
              </Link>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
