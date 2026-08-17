#!/usr/bin/env python3
"""Generate static illustrations for the CV business guide book.

Demonstrates: programmatic creation of consistent book visuals (diagrams, icons).
Expected behavior: writes PNG files to assets/illustrations/ with deterministic layout.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TOPIC_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = TOPIC_DIR / "assets" / "illustrations"

# Brand palette
NAVY = (26, 54, 93)
BLUE = (49, 130, 206)
TEAL = (56, 178, 172)
GREEN = (56, 161, 105)
ORANGE = (237, 137, 54)
PURPLE = (128, 90, 213)
GRAY = (113, 128, 150)
LIGHT = (247, 250, 252)
WHITE = (255, 255, 255)
DARK = (45, 55, 72)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _rounded_rect(draw: ImageDraw.ImageDraw, xy, radius: int, fill, outline=None, width: int = 2) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _arrow(draw: ImageDraw.ImageDraw, start, end, color=BLUE, width: int = 4) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 14
    p1 = (end[0] - head * math.cos(angle - 0.4), end[1] - head * math.sin(angle - 0.4))
    p2 = (end[0] - head * math.cos(angle + 0.4), end[1] - head * math.sin(angle + 0.4))
    draw.polygon([end, p1, p2], fill=color)


def _title_bar(draw: ImageDraw.ImageDraw, title: str, w: int) -> None:
    draw.rectangle([0, 0, w, 72], fill=NAVY)
    draw.text((32, 20), title, fill=WHITE, font=_font(28, bold=True))


def _save(img: Image.Image, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    img.save(path, "PNG", optimize=True)
    return path


def draw_cover_hero() -> Path:
    w, h = 1200, 675
    img = Image.new("RGB", (w, h), NAVY)
    draw = ImageDraw.Draw(img)
    for i in range(0, w, 40):
        draw.line([(i, 0), (i - 80, h)], fill=(35, 70, 110), width=1)
    draw.ellipse([780, 80, 1120, 420], fill=BLUE)
    draw.ellipse([820, 120, 1080, 380], outline=WHITE, width=3)
    draw.ellipse([900, 200, 940, 240], fill=WHITE)
    draw.ellipse([980, 200, 1020, 240], fill=WHITE)
    draw.arc([880, 250, 1040, 340], 10, 170, fill=WHITE, width=4)
    draw.text((64, 180), "Компьютерное", fill=WHITE, font=_font(52, bold=True))
    draw.text((64, 250), "зрение", fill=TEAL, font=_font(52, bold=True))
    draw.text((64, 340), "Вводное руководство для бизнеса", fill=(200, 220, 240), font=_font(24))
    return _save(img, "01_cover_hero")


def draw_human_vs_machine() -> Path:
    w, h = 1200, 560
    img = Image.new("RGB", (w, h), LIGHT)
    draw = ImageDraw.Draw(img)
    _title_bar(draw, "Человек и машина: два способа «видеть»", w)
    boxes = [
        (60, 120, 560, 500, "Человек", GREEN, [
            "Мгновенно понимает контекст",
            "Узнаёт лица и эмоции",
            "Адаптируется к новым ситуациям",
            "Устаёт и отвлекается",
        ]),
        (640, 120, 1140, 500, "Компьютерное зрение", BLUE, [
            "Анализирует миллионы кадров",
            "Не устаёт 24/7",
            "Фиксирует каждую деталь",
            "Нужны данные и настройка",
        ]),
    ]
    for x1, y1, x2, y2, title, color, lines in boxes:
        _rounded_rect(draw, (x1, y1, x2, y2), 16, WHITE, color, 3)
        draw.text((x1 + 24, y1 + 20), title, fill=color, font=_font(26, bold=True))
        for i, line in enumerate(lines):
            draw.text((x1 + 24, y1 + 70 + i * 36), f"• {line}", fill=DARK, font=_font(20))
    return _save(img, "02_human_vs_machine")


def draw_pixel_to_meaning() -> Path:
    w, h = 1200, 400
    img = Image.new("RGB", (w, h), LIGHT)
    draw = ImageDraw.Draw(img)
    _title_bar(draw, "От пикселей к смыслу", w)
    steps = [
        ("Камера", "Изображение\n(пиксели)", ORANGE),
        ("Модель", "Признаки\n(паттерны)", BLUE),
        ("Решение", "Действие\n(сигнал)", GREEN),
    ]
    x = 80
    for i, (label, desc, color) in enumerate(steps):
        cx = x + 140
        _rounded_rect(draw, (x, 130, x + 280, 340), 20, WHITE, color, 3)
        draw.text((x + 24, 150), label, fill=color, font=_font(24, bold=True))
        for j, line in enumerate(desc.split("\n")):
            draw.text((x + 24, 200 + j * 32), line, fill=DARK, font=_font(20))
        if i < len(steps) - 1:
            _arrow(draw, (x + 290, 235), (x + 350, 235))
        x += 360
    return _save(img, "03_pixel_to_meaning")


def draw_cv_tasks() -> Path:
    w, h = 1200, 620
    img = Image.new("RGB", (w, h), LIGHT)
    draw = ImageDraw.Draw(img)
    _title_bar(draw, "Основные задачи компьютерного зрения", w)
    tasks = [
        ("Классификация", "«Что это?»\nКошка / Собака", BLUE, 60),
        ("Детекция", "«Где объект?»\nРамки вокруг объектов", TEAL, 420),
        ("Сегментация", "«Какие пиксели?»\nКонтур объекта", PURPLE, 780),
        ("OCR", "«Какой текст?»\nРаспознавание надписей", ORANGE, 60),
        ("Трекинг", "«Куда движется?»\nСлежение за объектом", GREEN, 420),
        ("Аномалии", "«Всё ли в норме?»\nДефект / отклонение", (229, 62, 62), 780),
    ]
    for title, desc, color, x in tasks:
        y = 120 if x < 400 else 350
        _rounded_rect(draw, (x, y, x + 340, y + 200), 16, WHITE, color, 3)
        draw.text((x + 20, y + 16), title, fill=color, font=_font(22, bold=True))
        for j, line in enumerate(desc.split("\n")):
            draw.text((x + 20, y + 60 + j * 30), line, fill=DARK, font=_font(18))
    return _save(img, "04_cv_tasks")


def draw_pipeline() -> Path:
    w, h = 1200, 380
    img = Image.new("RGB", (w, h), LIGHT)
    draw = ImageDraw.Draw(img)
    _title_bar(draw, "Типичный конвейер системы CV", w)
    labels = ["Камера", "Обработка", "Модель ИИ", "Решение", "Интеграция"]
    x = 40
    for i, label in enumerate(labels):
        color = [ORANGE, GRAY, BLUE, GREEN, NAVY][i]
        _rounded_rect(draw, (x, 140, x + 200, 320), 14, WHITE, color, 2)
        draw.text((x + 20, 210), label, fill=color, font=_font(20, bold=True))
        if i < len(labels) - 1:
            _arrow(draw, (x + 210, 230), (x + 250, 230))
        x += 230
    return _save(img, "05_pipeline")


def draw_training_cycle() -> Path:
    w, h = 1200, 500
    img = Image.new("RGB", (w, h), LIGHT)
    draw = ImageDraw.Draw(img)
    _title_bar(draw, "Цикл: данные → обучение → внедрение", w)
    cx, cy, r = 600, 300, 160
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=BLUE, width=4)
    nodes = [
        (600, 120, "Данные\n(фото, видео)", ORANGE),
        (880, 300, "Обучение\nмодели", BLUE),
        (600, 480, "Тестирование\nи метрики", TEAL),
        (320, 300, "Внедрение\nв процесс", GREEN),
    ]
    for nx, ny, text, color in nodes:
        _rounded_rect(draw, (nx - 90, ny - 45, nx + 90, ny + 45), 12, WHITE, color, 2)
        for j, line in enumerate(text.split("\n")):
            draw.text((nx - 70, ny - 20 + j * 22), line, fill=color, font=_font(18, bold=True))
    return _save(img, "06_training_cycle")


def draw_sector_grid(name: str, title: str, items: list[tuple[str, str, tuple]]) -> Path:
    w, h = 1200, 640
    img = Image.new("RGB", (w, h), LIGHT)
    draw = ImageDraw.Draw(img)
    _title_bar(draw, title, w)
    positions = [(60, 110), (420, 110), (780, 110), (60, 350), (420, 350), (780, 350)]
    for (x, y), (t, desc, color) in zip(positions, items):
        _rounded_rect(draw, (x, y, x + 340, y + 210), 16, WHITE, color, 3)
        draw.text((x + 20, y + 16), t, fill=color, font=_font(22, bold=True))
        for j, line in enumerate(desc.split("\n")):
            draw.text((x + 20, y + 58 + j * 28), line, fill=DARK, font=_font(17))
    return _save(img, name)


def draw_implementation() -> Path:
    w, h = 1200, 480
    img = Image.new("RGB", (w, h), LIGHT)
    draw = ImageDraw.Draw(img)
    _title_bar(draw, "Дорожная карта внедрения CV", w)
    phases = [
        ("1. Гипотеза", "Где боль?\nКакой KPI?", ORANGE),
        ("2. Пилот", "Малый масштаб\n30–90 дней", BLUE),
        ("3. Масштаб", "Интеграция\nи мониторинг", TEAL),
        ("4. Развитие", "Новые сценарии\nи улучшения", GREEN),
    ]
    x = 60
    for title, desc, color in phases:
        _rounded_rect(draw, (x, 130, x + 250, 420), 16, WHITE, color, 3)
        draw.text((x + 20, 150), title, fill=color, font=_font(22, bold=True))
        for j, line in enumerate(desc.split("\n")):
            draw.text((x + 20, 210 + j * 32), line, fill=DARK, font=_font(18))
        if x < 900:
            _arrow(draw, (x + 260, 275), (x + 300, 275))
        x += 280
    return _save(img, "14_implementation")


def draw_ethics() -> Path:
    w, h = 1200, 480
    img = Image.new("RGB", (w, h), LIGHT)
    draw = ImageDraw.Draw(img)
    _title_bar(draw, "Этика и ответственное использование", w)
    items = [
        ("Приватность", "Минимизация данных,\nсогласие, анонимизация", BLUE),
        ("Справедливость", "Проверка на bias,\nравное качество", TEAL),
        ("Прозрачность", "Объяснимость решений,\nаудит", GREEN),
    ]
    x = 80
    for title, desc, color in items:
        _rounded_rect(draw, (x, 130, x + 320, 420), 16, WHITE, color, 3)
        draw.text((x + 24, 160), title, fill=color, font=_font(24, bold=True))
        for j, line in enumerate(desc.split("\n")):
            draw.text((x + 24, 220 + j * 32), line, fill=DARK, font=_font(20))
        x += 360
    return _save(img, "15_ethics")


def generate_all() -> list[Path]:
    paths = [
        draw_cover_hero(),
        draw_human_vs_machine(),
        draw_pixel_to_meaning(),
        draw_cv_tasks(),
        draw_pipeline(),
        draw_training_cycle(),
        draw_sector_grid(
            "07_industrial",
            "Промышленность: где CV приносит ценность",
            [
                ("Контроль качества", "Дефекты на линии\nбез остановки производства", BLUE),
                ("Безопасность труда", "Каски, зоны,\nопасное поведение", ORANGE),
                ("Подсчёт продукции", "Автоматический учёт\nна конвейере", GREEN),
                ("Предиктивное ТО", "Износ оборудования\nпо визуальным признакам", TEAL),
                ("Роботизация", "Навигация роботов\nна складе цеха", PURPLE),
                ("Документирование", "Фотоотчёты\nи traceability", GRAY),
            ],
        ),
        draw_sector_grid(
            "08_office",
            "Офис и корпоративные пространства",
            [
                ("Пропускной режим", "Face ID / QR +\nверификация", BLUE),
                ("Занятость переговорок", "Датчики + камеры\nаналитика", TEAL),
                ("Гибридный офис", "Загрузка рабочих мест\nheat maps", GREEN),
                ("Безопасность", "Сигнализация,\nнеавторизованный доступ", ORANGE),
                ("Комфорт", "Освещение, очереди\nв cafeteria", PURPLE),
                ("Соблюдение правил", "Дресс-код,\nзапретные зоны", GRAY),
            ],
        ),
        draw_sector_grid(
            "09_street",
            "Улица, город и инфраструктура",
            [
                ("Умные перекрёстки", "Потоки транспорта\nи пешеходов", BLUE),
                ("Парковка", "Свободные места,\nнарушения", TEAL),
                ("Экология", "Мусор, загрязнение\nводоёмов", GREEN),
                ("Безопасность", "Инциденты,\nтолпы, Vandalism", ORANGE),
                ("Дорожные работы", "Контроль техники\nи материалов", PURPLE),
                ("Ритейл у дома", "Витрины, очереди\nна улице", GRAY),
            ],
        ),
        draw_sector_grid(
            "10_retail",
            "Розничная торговля и e-commerce",
            [
                ("Self-checkout", "Сканирование\nбез кассира", BLUE),
                ("Планограммы", "Выкладка товара\nна полке", TEAL),
                ("Анти-theft", "Подозрительное\nповедение", ORANGE),
                ("Аналитика полок", "Out-of-stock\nалерты", GREEN),
                ("Примерочные", "Virtual try-on\nAR-зеркала", PURPLE),
                ("Склад", "Сортировка\nи комплектация", GRAY),
            ],
        ),
        draw_sector_grid(
            "11_healthcare",
            "Медицина и здравоохранение",
            [
                ("Диагностика", "Анализ снимков\n(MRI, КТ, рентген)", BLUE),
                ("Мониторинг", "Падения пациентов\nв палатах", TEAL),
                ("Хирургия", "Навигация\nи assist", GREEN),
                ("Лаборатории", "Подсчёт клеток\nмикроскопия", PURPLE),
                ("Документооборот", "OCR медкарт\nи рецептов", GRAY),
                ("Гигиена", "Соблюдение\nсанитарных норм", ORANGE),
            ],
        ),
        draw_sector_grid(
            "12_logistics",
            "Транспорт и логистика",
            [
                ("Сортировка", "Посылки на hub\nавтоматическая маршрутизация", BLUE),
                ("Склад", "Inventory\nrobot picking", TEAL),
                ("Автопарк", "ADAS, fatigue\nконтроль водителя", ORANGE),
                ("Порты", "Кontainer ID\nи damage check", GREEN),
                ("Last mile", "Дrones / robots\nдоставка", PURPLE),
                ("Трекинг", "GPS + video\nдоказательство", GRAY),
            ],
        ),
        draw_sector_grid(
            "13_security",
            "Безопасность и видеонаблюдение",
            [
                ("Perimeter", "Вторжение\nна объект", BLUE),
                ("Access control", "Биометрия\nи liveness", TEAL),
                ("Incident", "Дым, огонь,\nоружие", ORANGE),
                ("Crowd", "Плотность\nи эвакуация", GREEN),
                ("Forensics", "Поиск по\nappearance", PURPLE),
                ("Compliance", "Хранение\nи GDPR", GRAY),
            ],
        ),
        draw_implementation(),
        draw_ethics(),
    ]
    return paths


def main() -> None:
    paths = generate_all()
    print(f"Generated {len(paths)} illustrations in {OUT_DIR}")


if __name__ == "__main__":
    main()
