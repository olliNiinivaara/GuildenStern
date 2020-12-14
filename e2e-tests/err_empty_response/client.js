const { firefox } = require('playwright');

try {
  (async () => {
    const browser = await firefox.launch();
    const context = await browser.newContext();
    for (let i = 0; i < 20; i++) {
      console.log(i+1);
      const page = await context.newPage();
      await page.goto('http://127.0.0.1:8080/');
      const ok = await page.evaluate(() => {
        if (!document.getElementById("foo1")) return false
        if (!document.getElementById("foo2")) return false
        if (!document.getElementById("foo3")) return false
        return true
      });
      if (!ok) process.exit(-1 * i - 1)
    }
    console.log("err_empty_response test passed")
    await browser.close();    
  })();
} catch (e) {
  process.exit(-111)
}
