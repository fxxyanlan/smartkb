"""文档加载器：把 PDF / Markdown / Word / TXT 解析成纯文本。"""
from pathlib import Path

SUPPORTED_TYPES = {".pdf", ".md", ".markdown", ".docx", ".txt"}


def load_document(path: Path) -> str:
    """根据后缀选择合适的 loader，返回纯文本。"""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    elif ext in {".md", ".markdown"}:
        return _load_markdown(path)
    elif ext == ".docx":
        return _load_docx(path)
    elif ext == ".txt":
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"未实现的文件类型: {ext}")


def _load_pdf(path: Path) -> str:
    """PDF → 文本。"""
    import fitz

    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def _load_markdown(path: Path) -> str:
    """Markdown → 纯文本（保留段落结构）。"""
    return path.read_text(encoding="utf-8")


def _load_docx(path: Path) -> str:
    """Word → 文本。"""
    from docx import Document

    doc = Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs)
