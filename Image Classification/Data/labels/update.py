"""
A command-line utility to find and update objects within a JSON file based on a set of filter conditions. 
This is used for bulk-editing metadata, such as cleaning labels or grouping classes.

Example:
    python update.py <json_file> --filter "attr1=val1" "attr2!=val2" --update "attr3=new_val"
"""
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

def apply_filters(data, filters):
    """Returns a new list containing only the objects that match all filters."""
    filtered_data = []
    for obj in data:
        match = True
        for key, op, value in filters:
            if key not in obj:
                match = False
                break
            
            obj_val = str(obj[key])
            if op == "==" and obj_val != value:
                match = False
                break
            if op == "!=" and obj_val == value:
                match = False
                break
        
        if match:
            filtered_data.append(obj)
    
    return filtered_data

def update_objects(data, update_key, update_value):
    """Updates the attribute of each object in the provided list in-place."""
    update_count = 0
    for obj in data:
        if obj.get(update_key) != update_value:
            obj[update_key] = update_value
            update_count += 1
    return update_count

def main():
    """Main function to orchestrate the filtering and updating process."""
    parser = argparse.ArgumentParser(description="Filter and update objects in a JSON file.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file.")
    parser.add_argument("--filter", nargs='+', required=True, help="Filter conditions, e.g., 'type=car' 'color!=blue'.")
    parser.add_argument("--update", type=str, required=True, help="Update clause in 'attribute=value' format.")
    parser.add_argument("--dry-run", action='store_true', help="Show what would be changed without modifying the file.")
    args = parser.parse_args()

    try:
        update_key, update_value = args.update.split("=", 1)
    except ValueError:
        print(f"Error: Invalid update format '{args.update}'. Use 'attribute=value'.")
        return

    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filters = parse_filters(args.filter)
    filtered_data = apply_filters(data, filters)
    
    print(f"Found {len(filtered_data)} objects matching the filters.")

    if args.dry_run:
        print(f"Dry run enabled. Would update '{update_key}' to '{update_value}' for these objects.")
    else:
        update_count = update_objects(filtered_data, update_key.strip(), update_value.strip())
        with open(args.json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully updated {update_count} objects in '{args.json_file}'.")

if __name__ == "__main__":
    main()