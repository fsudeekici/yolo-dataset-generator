# YOLO Image Annotation Pipeline

This project creates YOLO-format annotations using images from a database and API.

## 1. Required Files Before Starting

Before you run the code, you must create two files:

- `id_list.txt`: List of image IDs (one per line)
- `image_list.txt`: List of file_name(one per line)

## 2. Folder Structure

.
- id_list.txt # (You must create this) Image IDs
- image_list.txt # (Optional) File names
- raw_images/ # Downloaded original images
- transaction_results/ # Data from database/API in JSON
- output_images/ # Images with bounding boxes
- output_yolo/ # YOLO label files (.txt)
- image_processor.py # Image conversion script
- main.py # Main Python script
- env # Your API and DB info
- .gitignore # Hides private files
- requirements.txt # Needed Python packages
- README.md # This file


## 3. How to Use

### Step 1 — Set your `.env` file:

```env
DB_HOST=your-db-host
DB_PORT=5432
DB_NAME=your-db-name
DB_USER=your-db-user
DB_PASSWORD=your-db-password

API_BASE_URL=https://your-api-url
LOGIN_EMAIL=your-email
LOGIN_PASSWORD=your-password
LOGIN_VERSION=1.0.0
LOGIN_OS=linux
