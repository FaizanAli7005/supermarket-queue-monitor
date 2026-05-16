import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


NAMES = {
    0: "person",
    1: "shopping_basket",
    2: "shopping_trolley",
}

COLORS = {
    "person": (255, 190, 60),
    "shopping_basket": (80, 220, 80),
    "shopping_trolley": (70, 130, 255),
    "estimated_customer": (220, 220, 220),
}

MEDIA_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
}


def read_roi(text, width, height):
    if not text:
        return 0, 0, width, height
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 4:
        raise ValueError("ROI must be x1,y1,x2,y2")
    if all(0 <= v <= 1 for v in values):
        x1, y1, x2, y2 = values
        return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
    return tuple(int(v) for v in values)


def center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def inside(box, roi):
    cx, cy = center(box)
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2


def order_key(customer, front):
    cx, cy = center(customer["box"])
    if front == "top":
        return cy
    if front == "bottom":
        return -cy
    if front == "left":
        return cx
    if front == "right":
        return -cx
    return cy


def distance(a, b):
    ax, ay = center(a)
    bx, by = center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def service_time(kind, basket_time, trolley_time, unknown_time):
    if kind == "shopping_basket":
        return basket_time
    if kind == "shopping_trolley":
        return trolley_time
    return unknown_time


def short_kind(kind):
    if kind == "shopping_basket":
        return "cart"
    if kind == "shopping_trolley":
        return "trolley"
    return "person"


def time_text(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    remaining = seconds % 60
    if remaining == 0:
        return f"{minutes}m"
    return f"{minutes}m {remaining}s"


def draw_text(frame, text, x, y, color, scale=0.65):
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def assign_items(people, items, width, height, link_ratio):
    free_items = items[:]
    limit = ((width * width + height * height) ** 0.5) * link_ratio
    for person in people:
        if not free_items:
            person["kind"] = "person"
            person["matched"] = None
            continue
        ranked = sorted(enumerate(free_items), key=lambda pair: distance(person["box"], pair[1]["box"]))
        index, item = ranked[0]
        if distance(person["box"], item["box"]) <= limit:
            person["kind"] = item["name"]
            person["matched"] = item
            free_items.pop(index)
        else:
            person["kind"] = "person"
            person["matched"] = None
    return people, free_items


def detections_from_frame(model, frame, args):
    result = model.predict(
        frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]

    height, width = frame.shape[:2]
    roi = read_roi(args.roi, width, height)
    people = []
    items = []

    for b in result.boxes:
        cls = int(b.cls[0])
        name = NAMES.get(cls, str(cls))
        conf = float(b.conf[0])
        box = [float(v) for v in b.xyxy[0].tolist()]
        if not inside(box, roi):
            continue
        entry = {"name": name, "conf": conf, "box": box}
        if name == "person":
            people.append(entry)
        elif name in ("shopping_basket", "shopping_trolley"):
            items.append(entry)

    people, unmatched = assign_items(people, items, width, height, args.link_distance)
    customers = people[:]

    for item in unmatched:
        customers.append({
            "name": "estimated_customer",
            "conf": item["conf"],
            "box": item["box"],
            "kind": item["name"],
            "matched": item,
        })

    customers.sort(key=lambda x: order_key(x, args.front))
    return customers, items, roi


def annotate(frame, customers, items, roi, args):
    waits = []
    total = 0
    basket_count = 0
    trolley_count = 0
    person_only_count = 0

    for customer in customers:
        kind = customer["kind"]
        waits.append(total)
        total += service_time(kind, args.basket_seconds, args.trolley_seconds, args.unknown_seconds)
        if kind == "shopping_basket":
            basket_count += 1
        elif kind == "shopping_trolley":
            trolley_count += 1
        else:
            person_only_count += 1

    rx1, ry1, rx2, ry2 = roi
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)

    for item in items:
        x1, y1, x2, y2 = [int(v) for v in item["box"]]
        color = COLORS[item["name"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_text(frame, f"{short_kind(item['name'])} {item['conf']:.2f}", x1, max(24, y1 - 8), color, 0.55)

    for i, customer in enumerate(customers):
        x1, y1, x2, y2 = [int(v) for v in customer["box"]]
        color = COLORS.get(customer["kind"], COLORS["estimated_customer"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        tag = f"#{i + 1} {short_kind(customer['kind'])} wait {time_text(waits[i])}"
        draw_text(frame, tag, x1, min(frame.shape[0] - 12, y2 + 24), color, 0.58)

    status = "LONG QUEUE" if len(customers) >= args.warning_count else "Queue OK"
    last_five = waits[-5:]
    last_five_text = ", ".join(time_text(x) for x in last_five) if last_five else "none"

    draw_text(frame, f"Customers: {len(customers)} | carts: {basket_count} | trolleys: {trolley_count} | person only: {person_only_count}", 18, 34, (255, 255, 255), 0.7)
    draw_text(frame, f"Last five waits: {last_five_text}", 18, 66, (255, 255, 255), 0.62)
    draw_text(frame, f"Estimated total service: {time_text(total)} | {status}", 18, 96, (0, 80, 255) if status == "LONG QUEUE" else (80, 220, 80), 0.68)

    return {
        "queue_length": len(customers),
        "basket_count": basket_count,
        "trolley_count": trolley_count,
        "person_only_count": person_only_count,
        "total_service_seconds": total,
        "last_customer_wait_seconds": waits[-1] if waits else 0,
        "last_five_waits": ";".join(str(int(x)) for x in last_five),
        "warning": status,
    }


def output_path_for(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return output_dir / f"{source.stem}_queue.png"
    return output_dir / f"{source.stem}_queue.mp4"


def process_image(model, source, output_dir, args):
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Could not read {source}")
    customers, items, roi = detections_from_frame(model, frame, args)
    stats = annotate(frame, customers, items, roi, args)
    output = output_path_for(source, output_dir)
    cv2.imwrite(str(output), frame)
    return output, [dict(file=source.name, frame=0, **stats)]


def process_video(model, source, output_dir, args):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    output = output_path_for(source, output_dir)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rows = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % args.frame_stride == 0:
            customers, items, roi = detections_from_frame(model, frame, args)
            last_stats = annotate(frame, customers, items, roi, args)
        else:
            last_stats = rows[-1] if rows else {
                "queue_length": 0,
                "basket_count": 0,
                "trolley_count": 0,
                "person_only_count": 0,
                "total_service_seconds": 0,
                "last_customer_wait_seconds": 0,
                "last_five_waits": "",
                "warning": "Queue OK",
            }
        writer.write(frame)
        rows.append(dict(file=source.name, frame=frame_index, **last_stats))
        frame_index += 1
    cap.release()
    writer.release()
    return output, rows


def collect_sources(source):
    source = Path(source)
    if source.is_file():
        return [source]
    files = []
    for path in source.iterdir():
        if path.suffix.lower() in MEDIA_EXTENSIONS and path.name.lower() not in ("results.png", "confusion_matrix.png", "labels.jpg"):
            files.append(path)
    return sorted(files, key=lambda x: x.name.lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--roi", default="")
    parser.add_argument("--front", choices=["top", "bottom", "left", "right"], default="top")
    parser.add_argument("--basket-seconds", type=int, default=45)
    parser.add_argument("--trolley-seconds", type=int, default=120)
    parser.add_argument("--unknown-seconds", type=int, default=75)
    parser.add_argument("--warning-count", type=int, default=6)
    parser.add_argument("--link-distance", type=float, default=0.24)
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()

    model = YOLO(args.model)
    output_dir = Path(args.output_dir)
    sources = collect_sources(args.source)
    all_rows = []

    for source in sources:
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            output, rows = process_image(model, source, output_dir, args)
        else:
            output, rows = process_video(model, source, output_dir, args)
        all_rows.extend(rows)
        print(f"saved {output}")

    csv_path = output_dir / "queue_estimates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["file", "frame", "queue_length", "basket_count", "trolley_count", "person_only_count", "total_service_seconds", "last_customer_wait_seconds", "last_five_waits", "warning"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
