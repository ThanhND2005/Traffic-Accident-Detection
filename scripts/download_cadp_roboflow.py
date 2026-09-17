"""
Download CADP dataset from Roboflow to datasets/cadp
"""

import os
import sys
import argparse

# Fix UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from roboflow import Roboflow


def download_cadp(api_key: str, output_dir: str = "datasets/cadp", version: int = 1):
    # Neu chua truyen API key, hoi truc tiep nguoi dung qua ban phim
    if not api_key or api_key.strip() in ["YOUR_ROBOFLOW_API_KEY", "YOUR_API_KEY", ""]:
        print("[*] Ban chua truyen API key qua tham so.")
        print("[*] Huong dan lay key mien phi (chi mat 1 phut):")
        print("    1. Dang nhap/tao tai khoan tai: https://app.roboflow.com")
        print("    2. Chon Settings (hoac vao https://app.roboflow.com/account/api)")
        print("    3. Copy 'Private API Key'")
        print("-" * 50)
        try:
            api_key = input("👉 Dan Roboflow API Key cua ban vao day va Enter: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[-] Da huy thao tac.")
            sys.exit(1)

    if not api_key or api_key.strip() in ["YOUR_ROBOFLOW_API_KEY", "YOUR_API_KEY", ""]:
        print("[-] Loi: API Key khong duoc de trong!")
        sys.exit(1)

    try:
        print("[*] Dang ket noi Roboflow voi API key...")
        rf = Roboflow(api_key=api_key)

        print(f"[*] Dang truy cap workspace 'yassine-pzpt7' - project 'cadp' (version {version})...")
        project = rf.workspace("yassine-pzpt7").project("cadp")

        os.makedirs(output_dir, exist_ok=True)

        print(f"[*] Dang tai dataset ve '{output_dir}' voi format YOLOv8...")
        dataset = project.version(version).download(model_format="yolov8", location=output_dir, overwrite=True)

        print(f"[+] Tai thanh cong! Du lieu duoc luu tai: {dataset.location}")
    except Exception as e:
        print(f"\n[-] Co loi xay ra khi tai dataset: {e}")
        if "401" in str(e) or "OAuthException" in str(e):
            print("[!] API Key khong hop le hoac bi sai. Hay kiem tra lai Private API Key tren Roboflow.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CADP dataset from Roboflow")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow Private API Key",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/cadp",
        help="Thu muc luu dataset (default: datasets/cadp)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Version dataset tren Roboflow (default: 1)",
    )

    args = parser.parse_args()
    download_cadp(api_key=args.api_key, output_dir=args.output_dir, version=args.version)
