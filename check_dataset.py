from config import DATASET_DIR


def main():
    print("=" * 60)
    print("CRACK500 DATASET CHECK")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print("ERROR: Dataset folder does not exist.")
        print(f"Path: {DATASET_DIR}")
        return

    jpg_files = {
        file.stem
        for file in DATASET_DIR.glob("*.jpg")
    }

    png_files = {
        file.stem
        for file in DATASET_DIR.glob("*.png")
    }

    matching = jpg_files & png_files
    jpg_without_png = jpg_files - png_files
    png_without_jpg = png_files - jpg_files

    print(f"Dataset folder: {DATASET_DIR}")
    print(f"JPG files: {len(jpg_files)}")
    print(f"PNG files: {len(png_files)}")
    print(f"Matching pairs: {len(matching)}")
    print(f"JPG without PNG: {len(jpg_without_png)}")
    print(f"PNG without JPG: {len(png_without_jpg)}")

    if (
        len(jpg_without_png) == 0
        and len(png_without_jpg) == 0
    ):
        print("\nDataset verification PASSED.")
        print("Every JPG has a matching PNG mask.")
    else:
        print("\nDataset verification FAILED.")
        print("Some images do not have matching masks.")


if __name__ == "__main__":
    main()