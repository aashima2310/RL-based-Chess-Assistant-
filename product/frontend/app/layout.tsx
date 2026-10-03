import "./globals.css";

export const metadata = {
  title: "ChessRL — AI Chess Coach",
  description: "Play, analyze, train and learn with your reinforcement-learning chess assistant.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
