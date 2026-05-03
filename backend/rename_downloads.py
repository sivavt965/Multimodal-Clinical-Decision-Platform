"""
rename_downloads.py
Scans your Downloads folder for the MIMIC-CXR .jpg files,
renames them to case_XXXXXXXX.png and copies them to the correct dicoms folder.

Usage:
    py -3.14 rename_downloads.py
    py -3.14 rename_downloads.py --src "C:/Users/yourname/Downloads"
"""

import argparse
import shutil
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
_DEST    = _BACKEND.parent / "frontend" / "public" / "mock-data" / "dicoms"

# dicom_filename_stem → [case_short_id, ...]  (some DICOMs map to 2 cases)
DICOM_TO_CASE: dict[str, list[str]] = {
    "ca2072de-c2407195-52a28692-909abc3f-ffdd05dc": ["3b1950dc"],
    "961ad519-28cb7c1e-cbffc899-895b7046-a3890eed": ["2d15ff35"],
    "8cf1576a-dc3d62b5-f479a641-93970390-8717d52a": ["38186792"],
    "38d11cf4-31201d6b-f761f181-41cb6e56-dd63e24e": ["f7148544"],
    "44dc1f43-cf5c869b-ce73b900-a075c0b7-bc62a0b9": ["e427a8fb"],
    "adaf646d-deb49f41-d8675cd2-949e5b8c-85ec4006": ["8343b48c"],
    "ea712bd4-a9219a81-79c83f99-b3569369-5156b499": ["ec4ee9ea", "07afb995"],
    "fce98979-0f2e775e-516d0141-7f0ea1a8-5a0c4979": ["60b43b71"],
    "98dd2a05-446f246b-4a6a0b24-51893821-c315cd2e": ["f0d86a7b", "1845969f"],
    "b85037ba-54d82db5-d8239b96-8aae46d0-ef478701": ["d71eef46"],
    "6907b6fb-52f9ff7c-c96374bd-1416e43b-2a5fae63": ["4abed709"],
    "b3c9ca9b-6913daee-e3fc00ef-bc48f947-5f225764": ["a67377ff"],
    "018aa67e-edbe363c-b7b7db2a-6cbf31b9-a55e9aae": ["9c6b22f7"],
    "f7e1ba52-abf13d1e-8b44b96e-dc65c6b5-6888fede": ["36fcd761"],
    "53e14f27-9d889ec5-b84ae3bb-9b61bccd-a0ed3d4c": ["7d8ce05a"],
    "7705678d-e97fd6af-95a3781d-29db3f83-4811df87": ["dbb0f050"],
    "fb95b0d2-2acbc31f-4ebaeecf-db8b421f-2aa3164f": ["7e31a8b0"],
    "184821c9-ca7e5c9d-e395c747-c93ed58e-f6016bc3": ["bfbd4450"],
    "ce749038-b3adc72f-4b6e74d6-0a49a8e4-b39363d4": ["27c45cfd"],
    "591d63de-c6207d4f-f03aa20d-bc4721f0-ec00cca1": ["e4300324"],
    "7c1980c0-9a1b764c-168b0395-8c37a6c4-b7f0db78": ["df7bea55"],
    "13651340-ba729357-685a3797-a0ce7ba3-8c28560c": ["4026462e"],
    "42846124-cd92a85f-29171bcb-1810fba0-e4b14742": ["e96d0a66"],
    "0621ed08-b94ab3c5-b49b96bf-08eda64b-478ccc43": ["aa25e1b2"],
    "3f3ff61f-ceb502d4-ad3354cf-ae6115ec-c84b0967": ["88008c93"],
    "db400c7d-5b44edc1-5a5bfbcf-5bb41da4-40834dff": ["7eb0348b"],
    "601aa5d2-2ece0ce9-7530c5e0-45158ff0-b5ba1aa9": ["f3597bf0"],
    "62f1d896-1ba64e45-6dea303e-ec84bef7-11e198f3":  ["6c247d24"],
    "a38861c4-d5aacacd-52df0234-66b01de1-11408d1b":  ["7e5f562d"],
    "19fb5b6b-d67165e9-812256b7-4994a3e0-75d960f8":  ["1ad15ac4"],
    "18752f1a-c05cb627-20a21fee-61938d89-d63fbffb":  ["d9dfe7a3"],
    "a3c32c0c-5e86e947-eeab36d9-7fc0fb8d-bb9922a1":  ["7037c41f"],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default=str(Path.home() / "Downloads"),
        help="Folder containing your downloaded .jpg files (default: ~/Downloads)",
    )
    args = parser.parse_args()

    src_dir = Path(args.src)
    if not src_dir.exists():
        print(f"Source folder not found: {src_dir}")
        return

    _DEST.mkdir(parents=True, exist_ok=True)

    jpg_files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg"))
    if not jpg_files:
        print(f"No .jpg files found in {src_dir}")
        return

    copied = 0
    skipped = 0
    unknown = []

    for jpg in jpg_files:
        stem = jpg.stem  # e.g. "ca2072de-c2407195-..."
        cases = DICOM_TO_CASE.get(stem)
        if not cases:
            unknown.append(jpg.name)
            continue

        for short_id in cases:
            dest = _DEST / f"case_{short_id}.png"
            shutil.copy2(jpg, dest)
            print(f"  {jpg.name}  ->  case_{short_id}.png")
            copied += 1

    print()
    print(f"Copied:  {copied} files")
    if skipped:
        print(f"Skipped: {skipped} (already existed)")
    if unknown:
        print(f"Unknown: {len(unknown)} files not in mapping:")
        for u in unknown:
            print(f"  {u}")

    # Show what's still missing
    all_needed = {s for cases in DICOM_TO_CASE.values() for s in cases}
    already_have = {"f1e3d197","ad58402a","2f54d7f5","813d610a","b5113ad5","238ecb55"}
    need = all_needed - already_have
    missing = [s for s in need if not (_DEST / f"case_{s}.png").exists()]

    if missing:
        print(f"\nStill missing {len(missing)} images:")
        for s in sorted(missing):
            print(f"  case_{s}.png")
        print("\nDownload them from download_urls.txt and re-run this script.")
    else:
        print("\nAll 34 images in place! Now run:")
        print("  py -3.14 trim_to_40.py")


if __name__ == "__main__":
    main()
