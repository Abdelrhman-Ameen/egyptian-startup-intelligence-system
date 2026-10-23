// Render the local report for layout inspection. Requires Playwright and Edge/Chromium.
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import fs from 'node:fs/promises';
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const require = createRequire(process.env.EDA_NODE_MODULES
  ? path.join(process.env.EDA_NODE_MODULES, 'package.json')
  : import.meta.url);
const { chromium } = require('playwright');
const report = path.join(root, 'reports/spain_technology_ventures/spain_ventures_eda_report.html');
const output = path.join(root, 'tmp/spain_report_preview');
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({channel: 'msedge', headless:true});
try {
  const page = await browser.newPage({viewport:{width:1400,height:1050},deviceScaleFactor:1});
  await page.goto(pathToFileURL(report).href);
  await page.screenshot({path:path.join(output,'overview.png')});
  for (const id of ['models','fit']) {
    await page.locator(`#${id}`).scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(output,`${id}.png`)});
  }
  const result = await page.evaluate(() => ({
    pageWidth: document.documentElement.clientWidth,
    contentWidth: document.documentElement.scrollWidth,
    images: document.images.length,
    brokenImages: [...document.images].filter(i=>!i.complete||i.naturalWidth===0).length,
    tableCount: document.querySelectorAll('table').length,
    title: document.title
  }));
  await fs.writeFile(path.join(output,'layout_checks.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
} finally { await browser.close(); }
