"""
A utility script to remove image files from a directory based on
metadata filters defined in a corresponding JSON file. This is useful for
culling a dataset of unwanted images (e.g., removing all images with
a specific attribute).

Usage:
    python delete.py <json_file> <image_directory> "attr1=val1" "attr2!=val2" ...
"""
import os
import json
import argparse

def parse_filters(filter_strings):
    """Parses filter strings into a list of (attribute, operator, value) tuples."""
    filters = []
    
    for f_str in filter_strings:
        
        if "!=" in f_str:
            key, value = f_str.split("!=", 1)
            filters.append((key.strip(), "!=", value.strip()))
        
        elif "=" in f_str:
            key, value = f_str.split("=", 1)
            filters.append((key.strip(), "==", value.strip()))
        
        else:
            raise ValueError(f"Invalid filter format: '{f_str}'. Use 'key=value' or 'key!=value'.")
    
    return filters

def matches_filters(obj, filters):
    """Checks if a single JSON object matches all provided filters."""
    for key, op, value in filters:
        
        if key not in obj:
            return False
        
        obj_val = str(obj[key])
        
        if op == "==" and obj_val != value:
            return False
        if op == "!=" and obj_val == value:
            return False
    
    return True

def filter_and_remove_images(json_path: str, directory: str, filters: List[Tuple[str, str, str]], dry_run: bool, confirm: bool) -> None:
    """Finds and removes image files that match the metadata filters."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    image_filenames_on_disk = set(os.listdir(directory))
    
    images_to_remove = [
        obj["filename"] for obj in data if obj["filename"] in image_filenames_on_disk and matches_filters(obj, filters)
    ]
    
    if not images_to_remove:
        print("No images found matching the specified filters.")
        return
    
    print(f"Found {len(images_to_remove)} images to be removed based on filters.")
    
    if dry_run:
        print("\n--- Dry Run Mode ---")
        print("The following files would be deleted:")
        
        for img_name in images_to_remove:
            print(f"  - {img_name}")
        
        return
    
    if confirm:
        proceed = input("Are you sure you want to permanently delete these files? (y/N): ").strip().lower()
        if proceed != "y":
            print("Operation cancelled by user.")
            return
    
    deleted_count = 0
    for img_name in images_to_remove:
        img_path = os.path.join(directory, img_name)
        
        try:
            os.remove(img_path)
            deleted_count += 1
        except OSError as e:
            print(f"Error removing file {img_path}: {e}")
    
    print(f"\nDeletion complete. Removed {deleted_count} files.")

def main():
    """Main function to orchestrate the image filtering and removal process."""
    parser = argparse.ArgumentParser(description="Filter and remove images based on JSON metadata.")
    parser.add_argument("json_file", help="Path to the JSON metadata file.")
    parser.add_argument("directory", help="Directory containing the image files.")
    parser.add_argument("filters", nargs="+", help="Filters to apply (e.g., 'attribute=value', 'attribute!=value').")
    parser.add_argument("--dry-run", action="store_true", help="Show which files would be deleted without actually deleting them.")
    parser.add_argument("--confirm", action="store_true", help="Prompt for confirmation before deleting files.")
    
    args = parser.parse_args()
    
    try:
        filters = parse_filters(args.filters)
        filter_and_remove_images(args.json_file, args.directory, filters, args.dry_run, args.confirm)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()