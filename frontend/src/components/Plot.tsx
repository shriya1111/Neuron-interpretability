"use client";

import { useEffect, useRef } from "react";
import { useLab } from "@/lib/state";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Props = {
  data: any[];
  layout?: Record<string, any>;
  height?: number;
  onClick?: (pt: { x: any; y: any; z?: any; pointIndex?: any; curveNumber: number }) => void;
};

/** Thin Plotly wrapper: loaded client-side only, theme-aware, click events forwarded. */
export default function Plot({ data, layout, height = 260, onClick }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const { dark } = useLab();
  const clickRef = useRef(onClick);
  clickRef.current = onClick;

  useEffect(() => {
    let cancelled = false;
    let el: any = null;
    (async () => {
      const Plotly: any = (await import("plotly.js-dist-min")).default;
      if (cancelled || !ref.current) return;
      el = ref.current;
      const fg = dark ? "#e4e6ea" : "#1b1c1e";
      const grid = dark ? "#2a2f37" : "#e2e2dc";
      const base = {
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: fg, size: 11, family: "ui-sans-serif, system-ui" },
        margin: { l: 48, r: 12, t: 24, b: 36 }, height,
        xaxis: { gridcolor: grid, zerolinecolor: grid, ...(layout?.xaxis ?? {}) },
        yaxis: { gridcolor: grid, zerolinecolor: grid, ...(layout?.yaxis ?? {}) },
        legend: { orientation: "h", y: -0.2 },
      };
      await Plotly.react(el, data, { ...base, ...layout, xaxis: base.xaxis, yaxis: base.yaxis },
        { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] });
      el.removeAllListeners?.("plotly_click");
      el.on?.("plotly_click", (ev: any) => {
        const p = ev.points?.[0];
        if (p && clickRef.current) clickRef.current({ x: p.x, y: p.y, z: p.z, pointIndex: p.pointIndex, curveNumber: p.curveNumber });
      });
    })();
    return () => { cancelled = true; };
  }, [data, layout, height, dark]);

  useEffect(() => () => {
    import("plotly.js-dist-min").then((m: any) => { if (ref.current) m.default.purge(ref.current); });
  }, []);

  return <div ref={ref} style={{ width: "100%", minHeight: height }} />;
}
