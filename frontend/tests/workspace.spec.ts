import { test, expect } from "@playwright/test";

test("cash planner computes a scenario, refreshes changed inputs, and resets", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /Give your next decision/ }),
  ).toBeVisible();
  await expect(page.locator(".metric-card").first()).toContainText("340,000");
  await page.getByLabel("Reduce payments").fill("0");
  await expect(
    page.getByText("Inputs changed.", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Compare cash paths" }).click();
  await expect(page.locator(".metric-card").first()).toContainText("-200,000");
  await page.getByRole("button", { name: "Load example" }).click();
  await expect(page.locator(".metric-card").first()).toContainText("340,000");
  await page.getByText("Inspect monthly cash movements").click();
  await expect(page.locator("tbody tr")).toHaveCount(12);
  await page.getByText("Inspect monthly cash movements").click();
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({
    path: "../docs/images/cash-planner.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("historical cases are evaluated for both financial schemas", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Financial ML lab" }).click();
  await expect(page.getByText("A case, not a claim.")).toBeVisible();
  await expect(
    page.getByLabel("Historical example").locator("option"),
  ).toHaveCount(6);
  await page.getByRole("button", { name: "Evaluate example" }).click();
  await expect(page.getByText("Observed historical outcome")).toBeVisible();
  await expect(page.locator(".score-number")).toHaveText(
    /<0\.001|>0\.999|0\.\d{3}/,
  );
  await page.getByRole("button", { name: "64 ratios Full schema" }).click();
  await page.getByRole("button", { name: "Evaluate example" }).click();
  await expect(page.getByText("Observed historical outcome")).toBeVisible();
  await page.screenshot({
    path: "../docs/images/research-lab.png",
    fullPage: true,
  });
});

test("methodology shows measured results and limits", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Methodology" }).click();
  await expect(
    page.getByRole("heading", { name: "The work behind the workspace." }),
  ).toBeVisible();
  await expect(page.locator(".comparison-panel tbody tr")).toHaveCount(14);
  await expect(
    page.getByText("PyTorch residual MLP", { exact: true }),
  ).toHaveCount(2);
});

test("mobile workspace fits the screen and supports navigation", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator(".metric-card").first()).toContainText("340,000");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "Financial ML lab" }).click();
  await expect(
    page.getByRole("button", { name: "Evaluate example" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "../docs/images/mobile-lab.png",
    fullPage: true,
  });
});
