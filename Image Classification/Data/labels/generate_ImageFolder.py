"""
Organizes an image dataset into a directory-per-class structure.
It reads a JSON metadata file, and for a specified attribute, creates
a subdirectory for each unique value of that attribute. It then creates
symbolic links to the original images in the corresponding subdirectory.

Note that images should be identified by the 'filename' attribute.

Example:
    python generate_ImageFolder.py <json_file> <output_base_dir> <source_image_dir> <attribute>
"""
import os
import json
import argparse
import re
from typing import List, Dict, Any

def create_symlinks_by_attribute(json_file, base_dir, source_dir, attribute):
    """Creates subdirectories and symbolic links based on a JSON attribute."""
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            objects = json.load(file)

        print(f"Organizing {len(objects)} images by attribute '{attribute}'...")
        
        link_count = 0
        for obj in objects:
            if 'filename' not in obj or attribute not in obj:
                continue

            filename = obj['filename']
            attribute_value = str(obj[attribute]).strip()
            
            # Sanitize the attribute value to create a valid directory name
            safe_subdir_name = re.sub(r'[^\w\-_\. ]', '_', attribute_value)
            if not safe_subdir_name:
                safe_subdir_name = "UNKNOWN"

            # Define paths
            output_subdir = os.path.join(base_dir, safe_subdir_name)
            source_file_path = os.path.join(source_dir, filename)
            symlink_path = os.path.join(output_subdir, filename)

            # Create subdirectory if it doesn't exist
            os.makedirs(output_subdir, exist_ok=True)
            
            # Create the symbolic link if source exists and link doesn't
            if os.path.exists(source_file_path) and not os.path.lexists(symlink_path):
                os.symlink(source_file_path, symlink_path)
                link_count += 1

        print(f"Process complete. Created {link_count} new symbolic links in subdirectories under '{base_dir}'.")

    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file}'.")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    """Main function to parse arguments and run the symlinking process."""
    parser = argparse.ArgumentParser(description="Create subdirectories and symbolic links based on a JSON attribute.")
    parser.add_argument("json_file", help="Path to the JSON file containing the list of objects.")
    parser.add_argument("base_dir", help="Base directory where subdirectories will be created.")
    parser.add_argument("source_dir", help="Directory where the source image files are located.")
    parser.add_argument("attribute", help="Attribute name from the JSON to use for creating subdirectories.")
    args = parser.parse_args()

    create_symlinks_by_attribute(args.json_file, args.base_dir, args.source_dir, args.attribute)

if __name__ == "__main__":
    main()