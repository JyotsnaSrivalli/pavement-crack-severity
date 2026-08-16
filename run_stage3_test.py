from pathlib import Path

from config import DATASET_DIR


TEST_COUNT = 10


def main():
    print("=" * 60)
    print("STAGE 3 DATASET TEST")
    print("=" * 60)

    jpg_files = sorted(
        DATASET_DIR.glob("*.jpg")
    )[:TEST_COUNT]

    if not jpg_files:
        print("ERROR: No JPG images found.")
        return

    successful = 0
    failed = 0

    for index, image_path in enumerate(
        jpg_files,
        start=1
    ):
        mask_path = (
            DATASET_DIR
            / f"{image_path.stem}.png"
        )

        print("\n" + "-" * 60)
        print(
            f"[{index}/{len(jpg_files)}] "
            f"{image_path.name}"
        )

        if not mask_path.exists():
            print("FAILED: Matching PNG mask not found.")
            failed += 1
            continue

        print(
            f"Mask: {mask_path.name}"
        )

        # At this stage we are only verifying
        # that the image-mask pair can be found.
        print("Pair found successfully.")

        successful += 1

    print("\n" + "=" * 60)
    print("STAGE 3 TEST SUMMARY")
    print("=" * 60)

    print(
        f"Images checked: {len(jpg_files)}"
    )

    print(
        f"Successful pairs: {successful}"
    )

    print(
        f"Failed pairs: {failed}"
    )

    if failed == 0:
        print(
            "\nStage 3 dataset pairing test PASSED."
        )
    else:
        print(
            "\nStage 3 dataset pairing test FAILED."
        )


if __name__ == "__main__":
    main()