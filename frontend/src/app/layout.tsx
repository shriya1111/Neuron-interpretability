import type { Metadata } from "next";
import "./globals.css";
import { LabProvider } from "@/lib/state";
import { Shell } from "@/components/Shell";

export const metadata: Metadata = {
  title: "NeuronLab",
  description: "Interactive mechanistic interpretability research instrument",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body>
        <LabProvider>
          <Shell>{children}</Shell>
        </LabProvider>
      </body>
    </html>
  );
}
