
from docling.document_converter import DocumentConverter


def main():
    # Change this to a local path or another URL if desired.
    # Note: using the default URL requires network access; if offline, provide a
    # local file path (e.g., Path("/path/to/file.pdf")).
    source = "https://arxiv.org/pdf/2408.09869"

    converter = DocumentConverter()
    result = converter.convert(source)

    # Print Markdown to stdout.
    result.document.save_as_markdown("output.md")
    print("Markdown saved to output.md")
    result.document.save_as_json("output.json")
    print("JSON saved to output.json")
    result.document.save_as_html("output.html")
    print("HTML saved to output.html")


if __name__ == "__main__":
    main()
