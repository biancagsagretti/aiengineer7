from pathlib import Path
import fitz
from docx import Document


def load_pdf(path: Path, image_dir: Path) -> list[dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            page_num = page_index + 1
            text = page.get_text("text") or ""
            records.append({
                "text": text,
                "metadata": {"document": path.name, "page": page_num, "content_type": "text"},
            })
            # Render de página completa: preserva diagramas vectoriales y escaneos.
            pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
            image_path = image_dir / f"{path.stem}_p{page_num}.png"
            pix.save(image_path)
            records.append({
                "text": "",
                "image_path": str(image_path),
                "metadata": {"document": path.name, "page": page_num, "content_type": "page_image"},
            })
            for img_index, image in enumerate(page.get_images(full=True)):
                xref = image[0]
                try:
                    extracted = pdf.extract_image(xref)
                    ext = extracted.get("ext", "png")
                    out = image_dir / f"{path.stem}_p{page_num}_img{img_index}.{ext}"
                    out.write_bytes(extracted["image"])
                    records.append({
                        "text": "",
                        "image_path": str(out),
                        "metadata": {"document": path.name, "page": page_num, "content_type": "embedded_image"},
                    })
                except Exception:
                    continue
    return records


def load_docx(path: Path, image_dir: Path) -> list[dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    doc = Document(path)
    records = []
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if text:
        records.append({"text": text, "metadata": {"document": path.name, "page": None, "content_type": "text"}})
    # Extrae imágenes embebidas del paquete DOCX. La página exacta no se infiere.
    for index, rel in enumerate(doc.part._rels.values()):
        if "image" in rel.reltype:
            try:
                blob = rel.target_part.blob
                content_type = rel.target_part.content_type
                ext = content_type.split("/")[-1].replace("jpeg", "jpg")
                image_path = image_dir / f"{path.stem}_image{index}.{ext}"
                image_path.write_bytes(blob)
                records.append({
                    "text": "",
                    "image_path": str(image_path),
                    "metadata": {"document": path.name, "page": None, "content_type": "embedded_image"},
                })
            except Exception:
                continue
    # Incluye tablas de Word como texto recuperable.
    for table_index, table in enumerate(doc.tables):
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        if rows:
            records.append({
                "text": "\n".join(rows),
                "metadata": {"document": path.name, "page": None, "content_type": "table", "table_index": table_index},
            })
    return records


def load_document(path: Path, image_dir: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, image_dir)
    if suffix == ".docx":
        return load_docx(path, image_dir)
    raise ValueError(f"Formato no soportado: {suffix}. Usá PDF o DOCX.")
