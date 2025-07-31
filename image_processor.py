# Import necessary libraries
import os
import ast
import json
import requests
import psycopg2
import psycopg2.extras
from PIL import Image, ImageOps, ImageDraw, ImageFont
import cv2
import unidecode
import numpy as np
from dotenv import load_dotenv

class ImageProcessor:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Define working directory and output folders
        self.WORKING_DIR = os.getcwd()
        self.DOWNLOAD_DIR = os.path.join(self.WORKING_DIR, "raw_images")
        self.RESULTS_DIR = os.path.join(self.WORKING_DIR, "transaction_results")
        self.OUTPUT_IMAGES_DIR = os.path.join(self.WORKING_DIR, "output_images")
        self.OUTPUT_YOLO_DIR = os.path.join(self.WORKING_DIR, "output_yolo")

        # Create directories if they don't already exist
        for directory in [self.DOWNLOAD_DIR, self.RESULTS_DIR, self.OUTPUT_IMAGES_DIR, self.OUTPUT_YOLO_DIR]:
            os.makedirs(directory, exist_ok=True)

        # Database connection parameters loaded from environment variables
        self.DB_PARAMS = {
            'host': os.getenv("DB_HOST"),
            'port': os.getenv("DB_PORT"),
            'dbname': os.getenv("DB_NAME"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD")
        }

        # API base URL and login credentials
        self.API_BASE_URL = os.getenv("API_BASE_URL")
        self.LOGIN_INFO = {
            'email': os.getenv("LOGIN_EMAIL"),
            'password': os.getenv("LOGIN_PASSWORD"),
            'version': os.getenv("LOGIN_VERSION"),
            'os': os.getenv("LOGIN_OS")
        }

        # Authentication token (will be set after login)
        self.login_token = None

    def connect_to_database(self):
        # Establish connection to the PostgreSQL database
        try:
            conn = psycopg2.connect(**self.DB_PARAMS)
            return conn
        except Exception as e:
            print(f"Database connection error: {e}")
            return None

    def extract_image_filenames_from_ids(self):
        # Step 1: Get image filenames corresponding to the transaction_log_ids from the database
        print("\n--- Step 1: Extract image filenames from IDs ---")
        try:
            with open("id_list.txt", "r") as f:
                id_list = [int(line.strip()) for line in f if line.strip().isdigit()]
        except FileNotFoundError:
            print("id_list.txt not found.")
            return False

        conn = self.connect_to_database()
        if not conn:
            return False

        try:
            with conn.cursor() as cursor, open("image_list.txt", "a") as output:
                for tid in id_list:
                    cursor.execute(
                        "SELECT file_name FROM your_schema.image_info WHERE transaction_log_id = %s", (tid,))
                    results = cursor.fetchall()
                    if results:
                        for r in results:
                            output.write(r[0] + "\n")
                            print(f"File name written for {tid}: {r[0]}")
                    else:
                        print(f"No result for ID {tid}")
            return True
        except Exception as e:
            print(f"Query error: {e}")
            return False
        finally:
            conn.close()

    def authenticate_and_download_images(self):
        # Step 2: Authenticate with the API and download images by filename
        print("\n--- Step 2: Authenticate and download images ---")
        login_url = f"{self.API_BASE_URL}/moblogin"
        try:
            response = requests.post(login_url, json=self.LOGIN_INFO)
            self.login_token = eval(response.text)['Content']
            print("Login successful.")
        except Exception as e:
            print(f"Login failed: {e}")
            return False

        try:
            with open("image_list.txt", 'r') as file:
                file_names = [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            print("image_list.txt not found.")
            return False

        for file_name in file_names:
            try:
                url = f"{self.API_BASE_URL}/getimagefile/{file_name}"
                resp = requests.get(url, headers={'token': self.login_token, 'project_id': '1'})

                if resp.status_code == 200:
                    save_path = os.path.join(self.DOWNLOAD_DIR, file_name)
                    with open(save_path, 'wb') as f:
                        f.write(resp.content)

                    # Ensure proper orientation using EXIF data
                    try:
                        im = Image.open(save_path)
                        im = ImageOps.exif_transpose(im)
                        im.save(save_path)
                        print(f"Downloaded and processed: {file_name}")
                    except Exception as img_err:
                        print(f"Image processing error: {file_name} – {img_err}")
                else:
                    print(f"Failed to download: {file_name}")
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")
        return True

    def fetch_response_data_from_database(self):
        # Step 3: Get the response_dicts (model output) from the database
        print("\n--- Step 3: Fetch response_dict from database ---")
        conn = self.connect_to_database()
        if not conn:
            return False

        try:
            with open("image_list.txt", 'r') as file:
                file_names = [line.strip() for line in file if line.strip()]

            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            seen_ids = set()

            for file_name in file_names:
                query = """
                    SELECT ii.file_name, tl.id AS transaction_log_id, tl.response_dict
                    FROM your_schema.image_info ii
                    JOIN your_schema.transaction_log tl ON ii.transaction_log_id = tl.id
                    WHERE ii.file_name = %s
                """
                cursor.execute(query, (file_name,))
                row = cursor.fetchone()

                if row:
                    tid = row['transaction_log_id']
                    if tid not in seen_ids:
                        path = os.path.join(self.RESULTS_DIR, f"{tid}_response.txt")
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(str(row['response_dict']))
                        seen_ids.add(tid)
                        print(f"Saved: {path}")
                else:
                    print(f"No record for {file_name}")

            cursor.close()
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
        finally:
            conn.close()

    def generate_yolo_annotations_with_visualization(self):
        # Step 4: Create YOLO annotation files and annotated images
        print("\n--- Step 4: Generate YOLO annotations with visualization ---")
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        conn = self.connect_to_database()
        if not conn:
            return False

        cursor = conn.cursor()

        try:
            for fname in os.listdir(self.RESULTS_DIR):
                path = os.path.join(self.RESULTS_DIR, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                        except:
                            data = ast.literal_eval(f.read())  # fallback if not valid JSON
                except Exception as e:
                    print(f"Error reading {fname}: {e}")
                    continue

                products = data.get("Content", {}).get("products") or data.get("Content", {}).get("Products")
                if not products:
                    continue

                for _, prod in products.items():
                    file_name = prod.get("File_Name")
                    output = prod.get("File_Inference_Output", {})
                    image_path = os.path.join(self.DOWNLOAD_DIR, file_name)

                    if not os.path.exists(image_path):
                        continue

                    img = cv2.imread(image_path)
                    if img is None:
                        continue

                    height, width = img.shape[:2]
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    overlay = Image.new("RGBA", pil_img.size)
                    draw = ImageDraw.Draw(overlay)
                    yolo_lines = []

                    for _, info in output.items():
                        if not all(k in info for k in ['label', 'x_min', 'x_max', 'y_min', 'y_max']):
                            continue

                        x_min = int(float(info['x_min']))
                        x_max = int(float(info['x_max']))
                        y_min = int(float(info['y_min']))
                        y_max = int(float(info['y_max']))

                        label = info.get("label")
                        class_id = 0  # Placeholder, should be mapped from label

                        # Draw bounding box and label
                        draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0, 255), width=2)
                        draw.text((x_min + 2, y_min + 2), label, font=font, fill=(255, 255, 255, 255))

                        # Convert box to YOLO format
                        x_c = ((x_min + x_max) / 2) / width
                        y_c = ((y_min + y_max) / 2) / height
                        w = (x_max - x_min) / width
                        h = (y_max - y_min) / height
                        yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

                    # Save annotated image
                    final = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
                    out_path = os.path.join(self.OUTPUT_IMAGES_DIR, f"annotated_{unidecode.unidecode(file_name)}")
                    cv2.imwrite(out_path, cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR))

                    # Save YOLO annotation
                    yolo_path = os.path.join(self.OUTPUT_YOLO_DIR, f"{os.path.splitext(file_name)[0]}.txt")
                    with open(yolo_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))

                    print(f"Saved annotated image and YOLO: {file_name}")
            return True
        except Exception as e:
            print(f"YOLO processing error: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def execute_complete_pipeline(self):
        # Execute all steps in order as a full pipeline
        steps = [
            (self.extract_image_filenames_from_ids, "Extract Filenames"),
            (self.authenticate_and_download_images, "Download Images"),
            (self.fetch_response_data_from_database, "Fetch DB Responses"),
            (self.generate_yolo_annotations_with_visualization, "Generate YOLO & Visualize")
        ]

        for func, name in steps:
            print(f"\n=== {name} ===")
            if not func():
                print(f"{name} failed.")
                return False
        print("\nAll steps completed successfully!")
        return True
