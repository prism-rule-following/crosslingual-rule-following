"""CAPABILITY_REPORT.md -> self-contained CAPABILITY_REPORT.html (images inlined)."""
import base64, os, re, markdown
H = os.path.dirname(os.path.abspath(__file__)); md = open(f"{H}/CAPABILITY_REPORT.md", encoding="utf-8").read()
body = markdown.markdown(md, extensions=["tables"])
body = re.sub(r'src="(data/[^"]+)"', lambda m: 'src="data:image/png;base64,' + base64.b64encode(open(f"{H}/{m.group(1)}", "rb").read()).decode() + '"', body)
css = "body{font:16px/1.55 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 16px;color:#202124}table{border-collapse:collapse;margin:1rem 0;font-size:14px}td,th{border:1px solid #dadce0;padding:5px 9px}th{background:#f1f3f4}img{max-width:100%}h1,h2{border-bottom:1px solid #eee;padding-bottom:4px}code{background:#f1f3f4;padding:1px 4px}"
open(f"{H}/CAPABILITY_REPORT.html", "w", encoding="utf-8").write(f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Capability baseline report</title><style>{css}</style>{body}")
print("CAPABILITY_REPORT.html written")
