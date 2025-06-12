import os
import csv
import logging

logger = logging.getLogger(__name__)


def append_row_to_csv(csv_path: str, row: dict) -> None:
    """Append a row of detection record to a CSV file, create file with header if not exists."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['file', 'label', 'score', 'url'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        logger.error(f"Failed to append row to CSV '{csv_path}': {e}")
        raise
