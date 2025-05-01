import contextlib, gzip, json, pickle, sqlite3
from io import StringIO

from sklearn.neighbors import KDTree


class dataLoader:
    def __init__(self, filename="", element_count=4):
        self.ELEMENT_COUNT = element_count
        assert filename, "No filename provided"
        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        with open(filename, "r") as file:
            header = file.readline().strip()
        self.ELEMENTS = [item.lower() for item in header.split("|")[: self.ELEMENT_COUNT]]

        columns = ", ".join([f"{element.lower()} REAL" for element in self.ELEMENTS]) + ", value TEXT"
        primary_keys = ", ".join(self.ELEMENTS)
        create_table_query = f"CREATE TABLE IF NOT EXISTS data (" f"{columns}, " f"PRIMARY KEY ({primary_keys})" ")"

        self.cursor.execute("DROP TABLE IF EXISTS data")
        self.cursor.execute(create_table_query)
        self.conn.commit()
        self.add_data(filename)

    def _parse_values(self, data: str):
        def is_float(value: str):
            try:
                float(value)
                return True
            except ValueError:
                return False

        json_result = {}
        parts = data.split("|")
        for key in ["T(K)", "Tsol+5", "Tsol-5"]:
            part = parts.pop(0)
            json_result[key] = {"phases": {}}
            while parts:
                part = parts.pop(0)
                if not part:
                    continue
                if is_float(part):  # if a value is a float means it's not a phase
                    break
                phase, mole_frac, _, volume_frac, *_ = part.split(",")
                phase_dict = {
                    phase: {
                        "mole_frac": mole_frac,
                        "volume_frac": volume_frac,
                    }
                }
                json_result[key]["phases"].update(phase_dict)
        # print(json.dumps(json_result, indent=4))
        return json.dumps(json_result)

    def add_data(self, filename: str):
        with open(filename, "r") as file:
            header = file.readline()
            file_elements = [item.lower() for item in header.split("|")[: self.ELEMENT_COUNT]]

            for element in file_elements:
                if element not in self.ELEMENTS:
                    raise ValueError(f"Element {element} not in {self.ELEMENTS} list")

            placeholders = ", ".join([f":{col}" for col in self.ELEMENTS] + [":value"])
            insert_query = f"INSERT OR IGNORE INTO data ({", ".join(self.ELEMENTS + ["value"])}) VALUES ({placeholders})"

            stream = StringIO(file.read(), newline=None)
            for line in stream:
                *elements, value = line.strip().split("|", self.ELEMENT_COUNT)
                record = dict(zip(self.ELEMENTS, elements))
                record.update({"value": self._parse_values(value)})
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
