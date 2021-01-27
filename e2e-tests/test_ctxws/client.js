const playwright = require('playwright');

(async () => {
  const browser = await playwright["chromium"].launch()
  const context = await browser.newContext()
  const page = await context.newPage()
  try {
    await page.goto('http://127.0.0.1:5050/')
    await page.waitForSelector("#li5")
    await page.setDefaultTimeout(100)
    await page.click('#wsbutton')
    await page.waitForTimeout(1000)
    await page.click('#closebutton')
  } catch(e) {
    console.log("ctxws test failed")
    process.exit(-5050)
  }
  console.log("ctxws test passed")
  await browser.close()
})()

