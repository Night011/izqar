"""Build the 8-slide IZQAR hackathon deck (PPTX) from verified material only.

Output: docs/presentation/IZQAR-hackathon.pptx
Every number here is copied from backend/models/*.metrics.json; screenshots are
real captures from docs/screenshots. Run:
    python scripts/build_presentation.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / 'docs' / 'screenshots'
OUT = ROOT / 'docs' / 'presentation' / 'IZQAR-hackathon.pptx'

BG = RGBColor(0x07, 0x13, 0x1B)
PANEL = RGBColor(0x0E, 0x22, 0x30)
PANEL_2 = RGBColor(0x12, 0x2B, 0x3A)
ACCENT = RGBColor(0x35, 0xD6, 0xC4)
ACCENT_DIM = RGBColor(0x14, 0x8A, 0x80)
TEXT = RGBColor(0xE9, 0xF5, 0xF3)
MUTED = RGBColor(0x92, 0xB4, 0xB1)
WARN = RGBColor(0xF2, 0xA5, 0x5C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = 'Arial'
W, H = 13.333, 7.5


def slide(prs):
    page = prs.slides.add_slide(prs.slide_layouts[6])
    bg = page.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H))
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    return page


def rect(page, x, y, w, h, color, line=None, radius=False):
    shape = page.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    shape.shadow.inherit = False
    if radius:
        shape.adjustments[0] = 0.08
    return shape


def text(page, x, y, w, h, value, size=18, bold=False, color=TEXT,
         align=PP_ALIGN.LEFT, spacing=1.05, anchor=MSO_ANCHOR.TOP):
    box = page.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    lines = value.split('\n')
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.line_spacing = spacing
        run = paragraph.add_run()
        run.text = line
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return box


def bullets(page, x, y, w, items, size=17, color=TEXT, gap=0.62, marker_color=ACCENT):
    for index, item in enumerate(items):
        top = y + index * gap
        dot = page.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(top + 0.09),
                                    Inches(0.09), Inches(0.09))
        dot.fill.solid()
        dot.fill.fore_color.rgb = marker_color
        dot.line.fill.background()
        dot.shadow.inherit = False
        text(page, x + 0.28, top, w - 0.28, gap, item, size=size, color=color, spacing=1.12)


def picture(page, path, x, y, w, h, border=True):
    """Place an image cropped to fill the box (keeps aspect ratio, no stretch)."""
    with Image.open(path) as image:
        iw, ih = image.size
    target = w / h
    source = iw / ih
    left = right = top = bottom = 0.0
    if source > target:                     # too wide: crop sides
        keep = target / source
        left = right = (1 - keep) / 2
    elif source < target:                   # too tall: crop top/bottom
        keep = source / target
        top = bottom = (1 - keep) / 2
    frame = rect(page, x - 0.03, y - 0.03, w + 0.06, h + 0.06,
                 ACCENT_DIM if border else BG)
    pic = page.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    pic.crop_left, pic.crop_right, pic.crop_top, pic.crop_bottom = left, right, top, bottom
    return frame, pic


def header(page, kicker, title, number, subtitle=None):
    text(page, 0.75, 0.45, 8.0, 0.3, kicker.upper(), size=11, bold=True, color=ACCENT)
    text(page, 0.75, 0.78, 11.6, 0.9, title, size=32, bold=True, color=TEXT, spacing=0.98)
    rule = rect(page, 0.75, 1.68, 1.1, 0.05, ACCENT)
    rule.shadow.inherit = False
    if subtitle:
        text(page, 0.75, 1.86, 11.6, 0.4, subtitle, size=15, color=MUTED)
    footer(page, number)


def footer(page, number):
    text(page, 0.75, 7.05, 6.0, 0.3, 'IZQAR · ITU Hackathon 2026', size=10, color=MUTED)
    text(page, 12.1, 7.05, 0.5, 0.3, f'{number:02d}', size=10, color=ACCENT,
         align=PP_ALIGN.RIGHT, bold=True)


def build():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)

    # ---------------------------------------------------------------- 1. Title
    page = slide(prs)
    picture(page, SHOTS / '01-dashboard.png', 7.05, 1.15, 5.55, 3.45)
    text(page, 0.85, 0.75, 2.0, 0.4, 'IZQAR', size=15, bold=True, color=ACCENT)
    text(page, 0.85, 1.25, 6.1, 2.7,
         'Интеллектуальная\nзащита финансовых\nопераций', size=40, bold=True, color=WHITE,
         spacing=0.96)
    rect(page, 0.85, 4.12, 1.4, 0.06, ACCENT)
    text(page, 0.85, 4.38, 5.9, 1.2,
         'Rule Engine, граф связей, ML-оценки и расследование — '
         'в одной консоли аналитика. Сигналы раздельны, решение остаётся за человеком.',
         size=16, color=MUTED, spacing=1.15)
    rect(page, 0.85, 5.95, 5.4, 0.02, PANEL_2)
    text(page, 0.85, 6.1, 5.4, 0.35, 'ITU Hackathon · 20 ноября 2026', size=14, color=TEXT, bold=True)
    text(page, 0.85, 6.45, 5.4, 0.35, 'research MVP · локальный CPU-инференс', size=13, color=MUTED)
    picture(page, SHOTS / '05-investigation-detail.png', 8.25, 5.05, 4.35, 1.85)
    text(page, 7.05, 4.72, 5.55, 0.3, 'Реальная консоль на демо-данных', size=11, color=MUTED)

    # -------------------------------------------------------------- 2. Problem
    page = slide(prs)
    header(page, 'Проблема', 'Одна операция — ещё не приговор, но и не «норма»', 2,
           'Аналитику нужно не число, а контекст и объяснимые причины решения.')
    bullets(page, 0.8, 2.35, 6.4, [
        'Скор без объяснения не годится для решения: непонятно, что его подняло.',
        'Ложные тревоги съедают время: очередь растёт быстрее, чем аналитики успевают смотреть.',
        'Схемы меняются: цепочки переводов, fan-in на один счёт, замкнутые циклы.',
        'Решения аналитика теряются между системами — нет журнала и происхождения оценки.',
    ], gap=0.85)
    cards = [('Без объяснений', 'Непонятно,\nпочему высокий риск'),
             ('Очередь', 'Ложные тревоги\nдороже пропусков'),
             ('Связи', 'Одиночная запись\nне видит схемы')]
    for index, (title, body) in enumerate(cards):
        top = 2.35 + index * 1.35
        rect(page, 7.6, top, 4.95, 1.15, PANEL, radius=True)
        text(page, 7.85, top + 0.16, 4.5, 0.3, title, size=14, bold=True, color=ACCENT)
        text(page, 7.85, top + 0.5, 4.5, 0.6, body, size=12.5, color=MUTED, spacing=1.05)

    # ------------------------------------------------------------- 3. Process
    page = slide(prs)
    header(page, 'Реальный процесс', 'От операции до решения аналитика', 3,
           'Пять шагов, которые можно показать на живом стенде за две минуты.')
    steps = [('01', 'Операция', 'сумма, получатель,\nустройство'),
             ('02', 'Rule Engine', 'объяснимые правила\nи уровень риска'),
             ('03', 'Граф связей', 'fan-in, цепочки,\nциклы'),
             ('04', 'Расследование', 'карточка, журнал,\nML-оценка'),
             ('05', 'Решение', 'статус и комментарий\nаналитика')]
    x0, width, gap = 0.75, 2.28, 0.24
    for index, (number, title, body) in enumerate(steps):
        x = x0 + index * (width + gap)
        rect(page, x, 2.5, width, 2.05, PANEL, radius=True)
        text(page, x + 0.22, 2.66, 1.0, 0.4, number, size=20, bold=True, color=ACCENT_DIM)
        text(page, x + 0.22, 3.12, width - 0.4, 0.4, title, size=15, bold=True, color=TEXT)
        text(page, x + 0.22, 3.58, width - 0.4, 0.9, body, size=12, color=MUTED, spacing=1.1)
        if index < 4:
            arrow = page.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + width + 0.045),
                                          Inches(3.55), Inches(0.16), Inches(0.26))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = ACCENT
            arrow.line.fill.background()
            arrow.shadow.inherit = False
    rect(page, 0.75, 4.95, 11.8, 0.9, PANEL_2, radius=True)
    text(page, 1.0, 5.12, 11.3, 0.55,
         'Rule Engine, ML-скор и графовый индикатор не склеиваются в выдуманную «вероятность '
         'мошенничества»: это разные источники сигналов с разными ограничениями.',
         size=14, color=TEXT, spacing=1.1)
    text(page, 0.75, 6.25, 11.8, 0.3,
         'Демо-операции синтетические; ML-модели обучаются и оцениваются отдельно на своих датасетах.',
         size=11, color=MUTED)

    # ------------------------------------------------------------ 4. Architecture
    page = slide(prs)
    header(page, 'Архитектура', 'Разделение источников сигналов', 4)
    blocks = [(0.8, 2.15, 3.2, 0.95, 'React + TypeScript', 'Vite · Cytoscape · demo-консоль', PANEL),
              (0.8, 3.35, 3.2, 0.95, 'FastAPI', 'Pydantic · SQLAlchemy · SQLite', PANEL),
              (0.8, 4.55, 3.2, 0.95, 'Rule Engine', 'объяснимые правила demo-потока', PANEL),
              (0.8, 5.75, 3.2, 0.95, 'NetworkX граф', 'направления, fan-in/out, циклы', PANEL),
              (4.75, 2.15, 3.5, 0.95, 'ML API', 'paysim · ieee · baf · ibm-aml · unified', PANEL),
              (4.75, 3.35, 3.5, 0.95, 'ML-артефакты', 'метаданные, версия, SHA-256', PANEL),
              (4.75, 4.55, 3.5, 0.95, 'Model Registry', 'статус, метрики, provenance', PANEL)]
    for x, y, w, h, title, body, color in blocks:
        rect(page, x, y, w, h, color, radius=True)
        text(page, x + 0.2, y + 0.14, w - 0.4, 0.3, title, size=14, bold=True, color=WHITE if color != PANEL else TEXT)
        text(page, x + 0.2, y + 0.48, w - 0.4, 0.4, body, size=10.5, color=MUTED, spacing=1.0)
    for y, label in ((2.62, 'HTTPS + X-API-Key'), (3.82, 'scores'), (5.02, 'leads'), (6.22, 'graph')):
        line = rect(page, 4.05, y, 0.62, 0.03, ACCENT_DIM)
        line.shadow.inherit = False
    bullets(page, 8.6, 2.2, 4.0, [
        'Нет артефакта — нет прогноза: endpoint отвечает 503.',
        'Каждая модель со своей схемой признаков и порогом.',
        'Порог выбирается только на validation.',
        'Всё считается локально на CPU, данные не уходят наружу.',
    ], size=12.5, gap=0.95)

    # ---------------------------------------------------------------- 5. Models
    build_models(prs)

    # --------------------------------------------------------------- 6. Metrics
    build_metrics(prs)

    # ------------------------------------------------------------------- 7. Demo
    page = slide(prs)
    header(page, 'Демонстрация', 'Что видно на экране', 7)
    picture(page, SHOTS / '02-ml-analytics.png', 0.75, 2.15, 6.0, 3.0)
    text(page, 0.75, 5.22, 6.0, 0.3, 'ML Analytics: метрики, пороги и версии моделей', size=11.5, color=MUTED)
    picture(page, SHOTS / '03-graph.png', 7.0, 2.15, 5.6, 2.1)
    text(page, 7.0, 4.32, 5.6, 0.3, 'Граф связей: fan-in, цепочки, циклы', size=11.5, color=MUTED)
    picture(page, SHOTS / '05-investigation-detail.png', 7.0, 4.72, 5.6, 1.9)
    text(page, 7.0, 6.68, 5.6, 0.3, 'Расследование: журнал, комментарий, ML-оценка', size=11.5, color=MUTED)

    # -------------------------------------------------------- 8. Limits/roadmap
    page = slide(prs)
    header(page, 'Ограничения и развитие', 'Что дальше и чего мы не обещаем', 8)
    text(page, 0.8, 2.2, 5.8, 0.35, 'ЧЕГО НЕТ', size=12, bold=True, color=WARN)
    bullets(page, 0.8, 2.65, 5.8, [
        'Research MVP: demo API-key, SQLite, без RBAC и неизменяемого аудита.',
        'Датасеты в основном синтетические; IEEE-CIS не доказывает качество на клиентах банка.',
        'Скоры не калиброваны как вероятности мошенничества.',
        'ML-артефакты не в Git: без файла модель честно «не установлена».',
        'CI и Docker не проверялись в этом релизе.',
    ], size=12.5, gap=0.78, marker_color=WARN)
    text(page, 7.0, 2.2, 5.6, 0.35, 'ЧТО ДАЛЬШЕ', size=12, bold=True, color=ACCENT)
    bullets(page, 7.0, 2.65, 5.6, [
        'Калибровка рисков на корректно выделенных данных.',
        'Потоковые счётчики и rolling-окно для бюджетного порога.',
        'Промоут проверенных кандидатов через отдельный релизный контур.',
        'Роли, аудит, мониторинг drift, Docker и CI.',
    ], size=12.5, gap=0.95)
    rect(page, 0.8, 6.2, 11.8, 0.6, PANEL_2, radius=True)
    text(page, 1.05, 6.35, 11.3, 0.3,
         'github.com/Night011/izqar · 162 backend-теста · 5 моделей в статусе ready',
         size=12.5, color=TEXT)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print('wrote', OUT, OUT.stat().st_size, 'bytes')


def build_models(prs):
    page = slide(prs)
    header(page, 'ML', 'Четыре задачи — четыре модели', 5)
    cards = [
        ('PaySim v5', 'синтетические переводы', '7 признаков', '955 744 test',
         'переводы: сумма и балансы'),
        ('IEEE-CIS v2', 'онлайн-платежи (Kaggle)', '45 признаков', '88 581 test',
         'карты, устройства, email'),
        ('BAF Base v1', 'синтетические заявки', '20 признаков', '96 843 test',
         'заявки на счёт'),
        ('IBM AML v1', 'синтетические AML-операции', '9 признаков', '762 103 test',
         'переводы между счетами'),
    ]
    for index, (name, dataset, features, rows, purpose) in enumerate(cards):
        x = 0.8 + (index % 2) * 6.0
        y = 2.05 + (index // 2) * 1.95
        rect(page, x, y, 5.7, 1.7, PANEL, radius=True)
        rect(page, x, y + 0.18, 0.07, 1.32, ACCENT)
        text(page, x + 0.3, y + 0.14, 5.1, 0.35, name, size=16, bold=True, color=TEXT)
        text(page, x + 0.3, y + 0.52, 5.1, 0.3, dataset, size=11.5, color=ACCENT)
        text(page, x + 0.3, y + 0.82, 5.1, 0.3, f'{features} · {rows}', size=11.5, color=MUTED)
        text(page, x + 0.3, y + 1.12, 5.1, 0.3, purpose, size=11.5, color=MUTED)
    rect(page, 0.8, 6.15, 11.8, 0.72, PANEL_2, radius=True)
    text(page, 1.05, 6.2, 11.3, 0.6,
         'Unified MTL v1 — экспериментальная multi-task сеть: общий backbone и четыре головы. '
         'Только validation-метрики, не замена специализированным моделям.',
         size=11.5, color=MUTED, spacing=1.05)


def build_metrics(prs):
    page = slide(prs)
    header(page, 'Результаты', 'Подтверждённые метрики на отложенном test', 6)
    columns = [('Модель', 0.8, 3.1), ('PR-AUC', 3.95, 1.25), ('F1', 5.25, 1.0),
               ('Recall', 6.3, 1.05), ('Precision', 7.4, 1.2),
               ('FP / 10k', 8.65, 1.15), ('Test', 9.85, 1.3), ('Данные', 11.2, 1.45)]
    y = 2.15
    rect(page, 0.75, y - 0.12, 11.9, 0.5, PANEL_2)
    for label, x, w in columns:
        text(page, x, y, w, 0.3, label, size=10.5, bold=True, color=ACCENT)
    rows = [
        ('PaySim v5', '0.9674', '0.8525', '0.7526', '0.9831', '0.54', '955 744', 'синтетика'),
        ('IEEE-CIS v2', '0.4442', '0.4642', '0.4149', '0.5270', '129.6', '88 581', 'историч. test'),
        ('BAF Base v1', '0.1997', '0.2411', '0.2073', '0.2882', '75.5', '96 843', 'синтетика'),
        ('IBM AML v1', '0.1013', '0.1153', '0.6675', '0.0631', '202.9', '762 103', 'синтетика'),
    ]
    for index, row in enumerate(rows):
        top = y + 0.55 + index * 0.72
        if index % 2 == 0:
            rect(page, 0.75, top - 0.1, 11.9, 0.66, PANEL)
        for (label, x, w), value in zip(columns, row):
            bold = label in ('Модель', 'PR-AUC')
            color = TEXT if label != 'Данные' else MUTED
            text(page, x, top, w, 0.3, value, size=11.5, bold=bold, color=color)
    rect(page, 0.75, 5.35, 11.9, 1.0, PANEL_2, radius=True)
    text(page, 1.0, 5.48, 11.4, 0.75,
         'Сравнивать строки между собой нельзя: это разные задачи и разные данные. '
         'IBM AML — слабое место по ложным тревогам; Unified MTL в таблицу не включён: '
         'у него только validation-метрики.',
         size=11.5, color=MUTED, spacing=1.15)
    text(page, 0.8, 6.6, 11.8, 0.3,
         'Скор — исследовательский, не калиброванная вероятность мошенничества.',
         size=11, color=WARN)


if __name__ == '__main__':
    build()
