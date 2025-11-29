
from google.cloud import vision_v1 as vision
from google.cloud import storage
from langchain_core.documents import Document
import json

def load_pdf_with_vision_ocr(input_gcs_uri, output_gcs_uri):
    client = vision.ImageAnnotatorClient()

    request = vision.AsyncAnnotateFileRequest(
        input_config=vision.InputConfig(
            gcs_source=vision.GcsSource(uri=input_gcs_uri),
            mime_type="application/pdf",
        ),
        features=[vision.Feature(type=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
        output_config=vision.OutputConfig(
            gcs_destination=vision.GcsDestination(uri=output_gcs_uri),
            batch_size=20,
        ),
    )

    operation = client.async_batch_annotate_files(requests=[request])
    print("⏳ Processing PDF OCR...")
    operation.result(timeout=300)

    print("✔ OCR complete! Fetching results...")
def read_ocr_output(output_gcs_uri):
    storage_client = storage.Client()
    bucket_name = output_gcs_uri.split("/")[2]
    prefix = "/".join(output_gcs_uri.split("/")[3:])

    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    full_text = ""

    for blob in blobs:
        data = json.loads(blob.download_as_bytes())
        responses = data["responses"]

        for page in responses:
            if "fullTextAnnotation" in page:
                full_text += page["fullTextAnnotation"]["text"]

    return full_text

