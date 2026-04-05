import ast
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    unique_categories = {}
    unique_locations = {}

    with open("/media/user/2TB/preprocessed_data.txt", "r") as f:
        count = 0
        lines = f.readlines()
        for line in lines:
            data = ast.literal_eval(line)
            categories = data["label"]
            if categories in unique_categories:
                unique_categories[categories] += 1
            else:
                print('New category found:', categories)
                unique_categories[categories] = 1
            location = data["location"]
            if location in unique_locations:
                unique_locations[location] += 1
            else:
                print('New location found:', location)
                unique_locations[location] = 1
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} lines out of {len(lines)} lines")
    
    # save unique categories and locations to json files
    with open("unique_categories.json", "w") as f:
        json.dump(unique_categories, f)
    with open("unique_locations.json", "w") as f:
        json.dump(unique_locations, f)
    
    # show top 20 categories
    sorted_categories = sorted(unique_categories.items(), key=lambda x: x[1], reverse=True)[:20]
    categories, counts = zip(*sorted_categories)
    plt.bar(categories, counts)
    plt.xticks(rotation=90)
    plt.title("Top 20 Categories")
    plt.xlabel("Categories")
    plt.ylabel("Counts")
    # save figure
    plt.savefig("top_categories.png")

    # show top 20 locations
    sorted_locations = sorted(unique_locations.items(), key=lambda x: x[1], reverse=True)[:20]
    locations, counts = zip(*sorted_locations)
    plt.bar(locations, counts)
    plt.xticks(rotation=90)
    plt.title("Top 20 Locations")
    plt.xlabel("Locations")
    plt.ylabel("Counts")
    # save figure
    plt.savefig("top_locations.png")
    