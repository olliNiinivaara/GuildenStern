const playwright = require('playwright');

(async () => {
  const browser = await playwright["chromium"].launch()
  const context = await browser.newContext()
  const page = await context.newPage()
  await page.goto('http://127.0.0.1:8070/')
  let content = await page.content()
  if (!content.includes("replyfromport8070")) {
    console.log("ctxheader test failed: 8070")
    process.exit(-8070)
  }
  await page.goto('http://127.0.0.1:8090/')
  content = await page.content()
  if (!content.includes("replyfromport8090")) {
    console.log("ctxheader test failed: 8090")
    process.exit(-8090)
  }
  await page.goto('http://127.0.0.1:8071/')
  content = await page.content()
  if (!content.includes("replyfromport8071")) {
    console.log("ctxheader test failed: 8071")
    process.exit(-8071)
  }
  await page.goto('http://127.0.0.1:8091/')
  content = await page.content()
  if (!content.includes("replyfromport8091")) {
    console.log("ctxheader test failed: 8091")
    process.exit(-8091)
  }
  await page.goto('http://127.0.0.1:8070/')
  content = await page.content()
  if (!content.includes("replyfromport8070")) {
    console.log("ctxheader test failed: 8070, second try")
    process.exit(-2)
  }
  await page.goto('http://127.0.0.1:8070/stop')  
  console.log("ctxheader test passed")
  await browser.close()
})()

