"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/** Client-side redirect that keeps ?key= so hosted access links work from the root URL. */
export default function Home() {
  const router = useRouter();
  useEffect(() => { router.replace("/atlas/qwen3-0.6b" + window.location.search); }, [router]);
  return null;
}
