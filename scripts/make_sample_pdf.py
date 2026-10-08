"""Creates data/sample/sample_policy.pdf (original, fictional text). Needs reportlab."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PAGES = [
    ["Acme Demo Company - Leave Policy (fictional sample)", "",
     "Casual leave: Employees receive 12 days of casual leave per calendar year.",
     "Sick leave: Employees receive 8 days of paid sick leave per year.",
     "Unused casual leave does not carry over to the next year."],
    ["Remote Work Guidelines (fictional sample)", "",
     "Employees may work remotely up to 2 days per week with manager approval.",
     "Remote employees must be reachable during core hours, 11:00 to 16:00.",
     "Company laptops must use the approved VPN when working remotely."],
]

out = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_policy.pdf"
c = canvas.Canvas(str(out), pagesize=A4)
for lines in PAGES:
    y = 780
    for line in lines:
        c.drawString(72, y, line)
        y -= 22
    c.showPage()
c.save()
print("wrote", out)
