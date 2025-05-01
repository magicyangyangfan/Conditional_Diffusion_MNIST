import contextlib, gzip, json, pickle, sqlite3
from io import StringIO

from sklearn.neighbors import KDTree


class dataLoader2:
    def __init__(self, filename="", element_count=4):
        self.ELEMENT_COUNT = element_count
        assert filename, "No filename provided"
        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        with open(filename, "r") as file:
            header = file.readline().strip()
        self.ELEMENTS = [item.lower() for item in header.split("\t")[: self.ELEMENT_COUNT]]
        self.PHASES = [item for item in header.split("\t")[self.ELEMENT_COUNT + 1:]]  # Skip Temperature(K)

        columns = ", ".join([f"{element.lower()} REAL" for element in self.ELEMENTS]) + ", value TEXT"
        primary_keys = ", ".join(self.ELEMENTS)
        create_table_query = f"CREATE TABLE IF NOT EXISTS data (" f"{columns}, " f"PRIMARY KEY ({primary_keys})" ")"

        self.cursor.execute("DROP TABLE IF EXISTS data")
        self.cursor.execute(create_table_query)
        self.conn.commit()
        self.add_data(filename)

    def _parse_values(self, data: list):
        """
        Parse phase amounts into a JSON format, excluding Temperature(K).
        """
        json_result = {"phases": {}}
        for phase, amount in zip(self.PHASES, data[1:]):  # Skip the first column (Temperature(K))
            json_result["phases"][phase] = float(amount)
        return json.dumps(json_result)

    def add_data(self, filename: str):
        with open(filename, "r") as file:
            header = file.readline()
            file_elements = [item.lower() for item in header.split("\t")[: self.ELEMENT_COUNT]]

            for element in file_elements:
                if element not in self.ELEMENTS:
                    raise ValueError(f"Element {element} not in {self.ELEMENTS} list")

            placeholders = ", ".join([f":{col}" for col in self.ELEMENTS] + [":value"])
            insert_query = f"INSERT OR IGNORE INTO data ({', '.join(self.ELEMENTS + ['value'])}) VALUES ({placeholders})"

            for line in file:
                parts = line.strip().split("\t")
                elements = parts[: self.ELEMENT_COUNT]
                phase_values = parts[self.ELEMENT_COUNT:]
                record = dict(zip(self.ELEMENTS, elements))
                record.update({"value": self._parse_values(phase_values)})
                with contextlib.suppress(sqlite3.IntegrityError):
                    self.cursor.execute(insert_query, record)
        self.conn.commit()
        # always rebuild tree after adding data
        self.cursor.execute("SELECT * FROM data")
        data = self.cursor.fetchall()

        self.composition_vectors = [item[: self.ELEMENT_COUNT] for item in data]
        self.values = [item[-1] for item in data]
        self.KDTREE = KDTree(self.composition_vectors, metric="euclidean")

    def export_data(self, filename="matQuery.pkl"):
        # use gzip to compress file, 200+MB -> 10+MB
        with gzip.GzipFile(filename, "wb") as f:
            pickle.dump((self.KDTREE, self.ELEMENTS, self.composition_vectors, self.values), f)
