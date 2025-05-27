# Install system dependencies (uncomment for Linux or Mac if needed)
# For Linux:
# !apt-get install poppler-utils tesseract-ocr libmagic-dev

# For Mac:
# !brew install poppler tesseract libmagic

# Install Python dependencies
%pip install -Uq "unstructured[all-docs]" pillow lxml
%pip install -Uq chromadb tiktoken
%pip install -Uq langchain langchain-community langchain-openai langchain-groq
%pip install -Uq python_dotenv

import os

# Set environment variables for API keys
# Replace "sk-..." with your actual API keys
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["GROQ_API_KEY"] = "sk-..."
os.environ["LANGCHAIN_API_KEY"] = "sk-..."
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from unstructured.partition.pdf import partition_pdf

output_path = "./content/"
# Ensure 'attention.pdf' is available in the specified path
file_path = output_path + 'attention.pdf'

# Partition the PDF into chunks, extracting tables and images
# Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
chunks = partition_pdf(
    filename=file_path,
    infer_table_structure=True,            # Extract tables
    strategy="hi_res",                     # Mandatory to infer tables for high-resolution processing

    extract_image_block_types=["Image"],   # Extract image blocks (add 'Table' to list to extract images of tables if needed)
    # image_output_dir_path=output_path,   # If None, images and tables will be saved in base64 within the payload

    extract_image_block_to_payload=True,   # If true, will extract base64 for API usage

    chunking_strategy="by_title",          # or 'basic'
    max_characters=10000,                  # Defaults to 500
    combine_text_under_n_chars=2000,       # Defaults to 0
    new_after_n_chars=6000,

    # extract_images_in_pdf=True,          # Deprecated parameter
)

# Check the types of elements obtained from the partition_pdf function
print(set([str(type(el)) for el in chunks]))

# Each CompositeElement contains a bunch of related elements.
# This makes it easy to use these elements together in a RAG pipeline.
# Example: Inspecting elements within the 4th chunk (index 3)
# Note: The exact index might vary depending on the PDF content.
elements = chunks[3].metadata.orig_elements
print(elements)

# This is what an extracted image looks like.
# It contains the base64 representation only because we set the param extract_image_block_to_payload=True

elements = chunks[3].metadata.orig_elements
chunk_images = [el for el in elements if 'Image' in str(type(el))]
# Print the dictionary representation of the first extracted image
print(chunk_images[0].to_dict())

# Separate tables from texts
tables = []
texts = []

for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)

    if "CompositeElement" in str(type((chunk))):
        texts.append(chunk)

# Get the images from the CompositeElement objects
def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

images = get_images_base64(chunks)

from IPython.display import HTML, display
from PIL import Image
import base64
from io import BytesIO



