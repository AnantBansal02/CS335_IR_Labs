from elasticsearch import Elasticsearch
import csv

# Create an Elasticsearch client instance
client = Elasticsearch([{'scheme':'http' ,'host': 'localhost', 'port': 9200}])

# Create an index
index_name = "my-email-collection"
index_settings = {
    "mappings": {
        "properties": {
            "date": {"type": "text",},
            "from": {"type": "text"},
            "to": {"type": "text"},
            "subject": {"type": "text"},
            "snippet": {"type": "text"}
        }
    }
}

client.indices.create(index=index_name, body=index_settings)

# Read and index documents from a CSV file
csv_file_path = "Gmail_export.csv"

with open(csv_file_path, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        document = {
            "date": row["date"],
            "from": row["from"],
            "to": row["to"],
            "subject": row["subject"],
            "snippet": row["snippet"]
        }
        client.index(index=index_name, body=document)

print("Data_Indexed_Successfully")
