# text_converter.py
import os
import re
import base64
from io import BytesIO
from PIL import Image
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_FILE_PATH = os.path.join(BASE_DIR, "dldg_databricks_1.md")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
CLEAN_MD_PATH = os.path.join(BASE_DIR, "dldg_databricks_1_clean.md")

os.makedirs(IMAGES_DIR, exist_ok=True)

def process_markdown_and_extract_images(md_path, output_md_path, images_dir):
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    pattern = r"!\[(.*?)\]\(data:image/(.*?);base64,(.*?)\)"
    matches = list(re.finditer(pattern, md_content))
    print(f"Found {len(matches)} images to extract.")

    clean_content = md_content
    for i, match in enumerate(matches):
        alt_text = match.group(1)
        img_format = match.group(2)
        base64_data = match.group(3)
        
        img_bytes = base64.b64decode(base64_data)
        image = Image.open(BytesIO(img_bytes))
        
        image_filename = f"image_{i+1}.{img_format if img_format else 'png'}"
        image_filepath = os.path.join(images_dir, image_filename)
        
        if image.mode in ("RGBA", "P") and img_format in ("jpeg", "jpg"):
            image = image.convert("RGB")
            
        image.save(image_filepath)
        
        relative_image_path = f"images/{image_filename}"
        replacement_string = f"![{alt_text}]({relative_image_path})"
        clean_content = clean_content.replace(match.group(0), replacement_string, 1)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(clean_content)
        
    return clean_content

# Generate the variables to be imported by the next script
print("Running text_converter.py: Cleaning markdown and generating nodes...")
cleaned_text = process_markdown_and_extract_images(MD_FILE_PATH, CLEAN_MD_PATH, IMAGES_DIR)
document = Document(text=cleaned_text, metadata={"source": "dldg_databricks_1.pdf"})
parser = MarkdownNodeParser()

# This 'nodes' variable is what we need for ChromaDB
nodes = parser.get_nodes_from_documents([document])
print(f"Successfully generated {len(nodes)} nodes ready for import.")
