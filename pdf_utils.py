from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text
