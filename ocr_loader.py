import fitz
import pytesseract
from PIL import Image
import io
from langchain_core.documents import Document


# Update this path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def load_pdf_with_ocr(path):
    pdf = fitz.open(path)
    docs = []

    for i in range(len(pdf)):
        page = pdf.load_page(i)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        text = pytesseract.image_to_string(img, lang="hin")  # Hindi OCR

        docs.append(
            Document(
                page_content=text,
                metadata={"source": path, "page": i}
            )
        )

    return docs
