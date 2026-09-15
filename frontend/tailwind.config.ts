import type { Config } from "tailwindcss";

/**
 * RAGForge design tokens.
 *
 * Identity: research workstation, not SaaS admin. Dark-first, hairline borders,
 * compact radii, mono data. Content = metadata: the accent is reserved for
 * signal, not decoration.
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Canvas — deep graphite with a navy undertone.
        canvas: "#0A0E14",
        // Elevated surfaces, in rising order.
        surface: {
          1: "#0E131B", // page section
          2: "#131924", // panel
          3: "#182030", // raised panel / input
          4: "#1E2836", // hover / active surface
        },
        // Hairline borders.
        line: {
          DEFAULT: "#1C2431",
          strong: "#2A3546",
          focus: "#3D4E66",
        },
        // Text hierarchy.
        ink: {
          DEFAULT: "#E7ECF3", // primary
          muted: "#93A0B4", // secondary
          faint: "#5D6B80", // tertiary / labels
          ghost: "#3D4859", // disabled / decorative
        },
        // Accents.
        accent: {
          DEFAULT: "#38BDF8", // cyan — interactive / retrieval
          soft: "#7DD3FC",
          dim: "#0B3B55",
        },
        violet: {
          DEFAULT: "#A78BFA", // experiments / knowledge structure
          dim: "#2E1D5E",
        },
        ok: {
          DEFAULT: "#34D399", // completed / connected
          dim: "#0A3D2C",
        },
        warn: {
          DEFAULT: "#FBBF24", // review / running / mock
          dim: "#4A3608",
        },
        bad: {
          DEFAULT: "#F87171", // errors only
          dim: "#4C1113",
        },
      },
      fontFamily: {
        sans: ['"Inter"', '"SF Pro Text"', "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ['"JetBrains Mono"', '"SF Mono"', "ui-monospace", "Menlo", "Consolas", "monospace"],
      },
      fontSize: {
        // Denser typographic scale for a data tool.
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
        xs: ["0.75rem", { lineHeight: "1.1rem" }],
        sm: ["0.8125rem", { lineHeight: "1.25rem" }],
        base: ["0.875rem", { lineHeight: "1.45rem" }],
      },
      borderRadius: {
        none: "0",
        sm: "2px",
        DEFAULT: "3px",
        md: "4px",
        lg: "6px",
      },
      boxShadow: {
        // Elevation is a hairline lift, not a glow.
        panel: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.6)",
        raise: "0 12px 32px -12px rgba(0,0,0,0.7)",
        focus: "0 0 0 1px #3D4E66",
      },
      keyframes: {
        "pulse-dot": {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.55", transform: "scale(0.82)" },
        },
        "flow": {
          "0%": { strokeDashoffset: "24" },
          "100%": { strokeDashoffset: "0" },
        },
        "shimmer": {
          "0%": { backgroundPosition: "-200px 0" },
          "100%": { backgroundPosition: "calc(200px + 100%) 0" },
        },
      },
      animation: {
        "pulse-dot": "pulse-dot 1.6s ease-in-out infinite",
        "flow": "flow 1.2s linear infinite",
        "shimmer": "shimmer 1.4s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
