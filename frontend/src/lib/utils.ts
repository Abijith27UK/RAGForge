import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** 0.8123 -> "0.812"; null -> "—" */
export function fmtScore(v: number | null | undefined, digits = 3): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

/** 0.964 -> "96.4%"; null -> "—" */
export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return "—";
  return (v * 100).toFixed(digits) + "%";
}

/** 1234567 -> "1.23M" */
export function fmtCount(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "k";
  return String(n);
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}
