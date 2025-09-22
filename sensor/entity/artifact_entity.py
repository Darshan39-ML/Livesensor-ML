from dataclasses import dataclass

@dataclass

class DataIngestionArtifact:
    trainig_file_path: str
    test_file_path: str