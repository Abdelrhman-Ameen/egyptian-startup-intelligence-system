import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  workers: 1,
  retries: 0,
  use: {
    baseURL: process.env.BASE_URL || "http://127.0.0.1:8000",
    headless: true,
    channel: process.env.CI ? undefined : "msedge",
    screenshot: "only-on-failure",
  },
  reporter: [["list"]],
});
