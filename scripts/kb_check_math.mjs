#!/usr/bin/env node
// Проверка LaTeX-формул в Markdown книги парсером KaTeX + правила GitHub для inline-математики.
// Использование (из корня репозитория, katex не входит в зависимости проекта):
//   npm i --no-save katex && node scripts/kb_check_math.mjs topics docs
// Код выхода 1, если найдены проблемы.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(path.join(process.cwd(), "/"));
let katex;
try { katex = require("katex"); } catch { console.error("katex не найден: выполните `npm i --no-save katex` в текущем каталоге"); process.exit(2); }

const roots = process.argv.slice(2).length ? process.argv.slice(2) : ["topics"];
const files = [];
const walk = (d) => { for (const e of fs.readdirSync(d, { withFileTypes: true })) { const p = path.join(d, e.name); if (e.isDirectory()) { if (!/node_modules|\.git/.test(p)) walk(p); } else if (e.name.endsWith(".md")) files.push(p); } };
for (const r of roots) fs.statSync(r).isDirectory() ? walk(r) : files.push(r);

let issues = 0;
const report = (file, line, src, msg) => { issues++; console.log(`${file}:${line}: ${src.replace(/\s+/g, " ").slice(0, 100)}\n    -> ${msg}`); };
const check = (file, line, src, display) => {
  try { katex.renderToString(src, { throwOnError: true, displayMode: display, strict: "ignore" }); }
  catch (e) { report(file, line, (display ? "$$ " : "$ ") + src, String(e.message).split("\n")[0].slice(0, 160)); }
};
for (const f of files) {
  const lines = fs.readFileSync(f, "utf8").split("\n");
  let inCode = false, inMath = false, buf = [], start = 0;
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (/^\s*(```|~~~)/.test(l)) { inCode = !inCode; continue; }
    if (inCode) continue;
    if (inMath) { if (l.trim() === "$$" || l.trim().endsWith("$$")) { buf.push(l.replace(/\$\$\s*$/, "")); check(f, start, buf.join("\n"), true); inMath = false; buf = []; } else buf.push(l); continue; }
    if (l.trim() === "$$") { inMath = true; start = i + 1; buf = []; continue; }
    if (/^\s*\$\$.+\$\$\s*$/.test(l)) { check(f, i + 1, l.trim().slice(2, -2), true); continue; }
    if (/^\s*\$\$/.test(l)) { inMath = true; start = i + 1; buf = [l.trim().slice(2)]; continue; }
    const re = /(^|[^\\$])\$([^$\n]+?)\$(?!\$)/g; let m;
    while ((m = re.exec(l))) {
      const src = m[2];
      if (/^\s|\s$/.test(src)) { report(f, i + 1, "$ " + src, "пробел сразу после/перед $ — GitHub не рендерит такую inline-формулу"); continue; }
      if (/\*\{/.test(src)) report(f, i + 1, "$ " + src, "подозрительный '*{' (вероятно, '_' заменён на '*')");
      check(f, i + 1, src, false);
    }
    const dollars = (l.match(/(^|[^\\])\$/g) || []).length;
    if (dollars % 2 === 1 && !/\$\$/.test(l)) report(f, i + 1, l.trim(), "нечётное число '$' в строке: либо незакрытая формула, либо знак доллара в тексте (экранируйте \\$)");
  }
  if (inMath) report(f, start, buf.join(" "), "незакрытый блок $$");
}
console.log(`\n${issues} issue(s) in ${files.length} md files`);
process.exit(issues ? 1 : 0);
