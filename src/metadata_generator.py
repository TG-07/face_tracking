import json
import csv

class MetadataGenerator:
    @staticmethod
    def generate_json_metadata(clip_metadata, output_file):
        with open(output_file, 'w') as f:
            json.dump(clip_metadata, f, indent=2)

    @staticmethod
    def generate_csv_metadata(clip_metadata, output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "start_time", "end_time", "face_coordinates"])
            writer.writeheader()
            for clip in clip_metadata:
                for frame in clip:
                    writer.writerow({
                        "file_name": frame["file_name"],
                        "start_time": frame.get("start_time", ""),
                        "end_time": frame.get("end_time", ""),
                        "face_coordinates": str(frame["face_coordinates"])
                    })

    @staticmethod
    def generate_metadata(clip_metadata, output_file, format='json'):
        if format.lower() == 'json':
            MetadataGenerator.generate_json_metadata(clip_metadata, output_file)
        elif format.lower() == 'csv':
            MetadataGenerator.generate_csv_metadata(clip_metadata, output_file)
        else:
            raise ValueError("Unsupported metadata format. Use 'json' or 'csv'.")
