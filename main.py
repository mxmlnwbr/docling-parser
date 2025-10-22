
from docling.document_converter import DocumentConverter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    AcceleratorDevice,
    TesseractOcrOptions,
    TableFormerMode,
    granite_picture_description,
)

import torch
from pathlib import Path



def define_options():
    if torch.cuda.is_available():
        device = "cuda"
        accelerator_device = AcceleratorDevice.CUDA
        print("Using CUDA device for AI models")
    elif torch.backends.mps.is_available():
        device = "mps"
        accelerator_device = AcceleratorDevice.MPS
        print("Using MPS device for AI models")
    else:
        device = "cpu"
        accelerator_device = AcceleratorDevice.CPU
        print("Using CPU device for AI models")

    # Configure accelerator options with detected device and Flash Attention 2
    accelerator_options = AcceleratorOptions(
        num_threads=4, 
        device=accelerator_device,
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options

    # OCR configuration
    pipeline_options.do_ocr = False
    pipeline_options.ocr_options = TesseractOcrOptions()

    # Formula and code enrichment
    pipeline_options.do_formula_enrichment = True
    pipeline_options.do_code_enrichment = True

    # Table structure recognition with device configuration
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True

    # Image generation
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.images_scale = 2.0

    # Picture classification and description with device configuration
    pipeline_options.do_picture_classification = True
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = granite_picture_description

    pipeline_options.picture_description_options.prompt = """What is on this image?"""

    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    })
    return converter


def main():
    # Change this to a local path or another URL if desired.
    # Note: using the default URL requires network access; if offline, provide a
    # local file path (e.g., Path("/path/to/file.pdf")).
    source = "https://arxiv.org/pdf/2408.09869"

    converter = define_options()
    result = converter.convert(source)

    # Create output Folder if it not exists
    Path("output").mkdir(exist_ok=True)

    # Print Markdown to stdout.
    result.document.save_as_markdown("output.md")
    print("Markdown saved to output.md")
    result.document.save_as_json("output.json")
    print("JSON saved to output.json")


if __name__ == "__main__":
    main()
