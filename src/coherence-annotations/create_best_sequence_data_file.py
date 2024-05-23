import json
import csv
import os

BUCKET_NAME = "coherence-annotation-images"

HEADER = ["recipe_id", "image_url", "title"]


# @pre: full_image_path.endswith(".png")
def get_url(full_image_path: str) -> str:
    return f"https://{BUCKET_NAME}.s3.amazonaws.com/{full_image_path}"


def create_file(image_dir, bucket_inner_path, recipes, result_file):
    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for img in os.listdir(image_dir):
            recipe_id = img.split(".")[0]
            title = recipes[recipe_id]["recipe"]["name"]
            url = get_url(os.path.join(bucket_inner_path, img))
            row = [recipe_id, url, title]
            writer.writerow(row)

if __name__ == "__main__":
    bucket_inner_path = "our-method-vs-ground-truth"
    dir_path = ""
    CSV_RESULT_FILE = os.path.join("", f"{bucket_inner_path}.csv")

    RECIPES_FILE = '.json'

    with open(RECIPES_FILE, "r") as f:
        recipes = json.load(f)

    create_file(dir_path, bucket_inner_path, recipes, CSV_RESULT_FILE)
