import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Cloudflare Pages + standalone output
  output: "export",
  basePath: "",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  // The frontend is static; all API calls go to the backend URL.
  // Set NEXT_PUBLIC_BACKEND_URL in your hosting environment.
  async rewrites() {
    return [];
  },
};

export default nextConfig;
