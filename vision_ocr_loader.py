from google.cloud import vision_v1 as vision
from google.cloud import storage
from langchain_core.documents import Document
import json


def extract_filename(gcs_uri: str) -> str:
    return gcs_uri.split("/")[-1]


def cache_blob_path(filename: str) -> str:
    return f"ocr-cache/{filename}.txt"


def check_cache(bucket_name: str, filename: str) -> bool:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(cache_blob_path(filename))
    return blob.exists()


def save_cache(bucket_name: str, filename: str, text: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(cache_blob_path(filename))
    blob.upload_from_string(text, content_type="text/plain")
    print(f"Cached OCR saved at: gs://{bucket_name}/{cache_blob_path(filename)}")


def read_cache(bucket_name: str, filename: str) -> str:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(cache_blob_path(filename))
    print(f"Using cached OCR for: {filename}")
    return blob.download_as_text()


def load_pdf_with_vision_ocr(input_gcs_uri: str, output_gcs_uri: str) -> str:

    bucket_name = input_gcs_uri.split("/")[2]
    filename = extract_filename(input_gcs_uri)

    if check_cache(bucket_name, filename):
        return read_cache(bucket_name, filename)
    
    print(f" No OCR cache found for {filename} — running OCR...")
    client = vision.ImageAnnotatorClient()

    request = vision.AsyncAnnotateFileRequest(
        input_config=vision.InputConfig(
            gcs_source=vision.GcsSource(uri=input_gcs_uri),
            mime_type="application/pdf",
        ),
        features=[
            vision.Feature(
                type=vision.Feature.Type.DOCUMENT_TEXT_DETECTION
            )
        ],
        output_config=vision.OutputConfig(
            gcs_destination=vision.GcsDestination(uri=output_gcs_uri),
            batch_size=20,
        ),
    )

    operation = client.async_batch_annotate_files(requests=[request])
    operation.result(timeout=300)
    print("OCR complete — fetching output JSON from GCS...")

    ocr_text = read_ocr_output(output_gcs_uri)

    save_cache(bucket_name, filename, ocr_text)

    return ocr_text


def read_ocr_output(output_gcs_uri: str) -> str:
    storage_client = storage.Client()

    bucket_name = output_gcs_uri.split("/")[2]
    prefix = "/".join(output_gcs_uri.split("/")[3:])

    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    full_text = ""

    for blob in blobs:
        raw = json.loads(blob.download_as_bytes())
        responses = raw.get("responses", [])

        for res in responses:
            annotation = res.get("fullTextAnnotation")
            if annotation:
                full_text += annotation.get("text", "")

    return full_text
