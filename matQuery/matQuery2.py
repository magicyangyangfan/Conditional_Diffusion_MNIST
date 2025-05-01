import gzip, pickle, numpy, json, re
from typing import Any, Dict, List
from itertools import combinations
import os


class matQuery:
    def __init__(self, filename="matQuery.pkl", use_resultant=True):
        self.KDTREE, self.ELEMENTS, self.composition_vectors, self.values = self.load_data(filename)
        self.use_resultant = use_resultant

    def query(self, params: Dict[str, Any], k: int = 5) -> str:
        # convert all keys to lower case to match with column names
        params = {k.lower(): v for k, v in params.items()}

        if not numpy.isclose(sum(params.get(element, 0) for element in self.ELEMENTS), 1, atol=0.001):
            return "The sum of all element compositions must be close to 1 (within a tolerance of 0.001)"

        query_vector = tuple([params.get(element, 0) for element in self.ELEMENTS])

        knn_rows = self._get_knn_values(params, k) #get the k nearest neighbors
        # print("knn_rows", knn_rows)
        selected_values, weights = (
            self._vectorial_resultant(knn_rows, query_vector, k)
            if self.use_resultant
            else self._naive_query(knn_rows, k)
        )
        # print(weights)
        result = self._get_weighted_result(selected_values, weights)

        return json.dumps(result, indent=4)

    def _get_weighted_result(self, values: List[str], weights: List[float]) -> Dict[str, Any]:
        json_data = [json.loads(js) for js in values] #{phase : {Al2Cu:0.022}... }
        result = {}
        
        # phasesSet = set()
        # phases = json_data[0]["phases"]
        # print(values)
        for data in json_data:
            for phase in data['phases'].keys():
                if phase not in result:
                    result[phase] = 0.0
                result[phase] += round(data['phases'][phase] * weights[json_data.index(data)], 4)
        # print(result)
        return result

    def _naive_query(self, rows: List, k: int) -> tuple[List[str], List[float]]:
        selected_values = [rows[i][-2] for i in range(k)]
        weights = (
            [1 / k] * k
        )
        return selected_values, weights

    def _vectorial_resultant(self, rows: List, query_vector: tuple, k: int) -> tuple[List[str], List[float]]:
        best_subset = None
        min_vector_sum_length = float("inf")
        for subset_size in range(1, len(rows) + 1):
            for subset in combinations(range(k), subset_size):
                resultant_vector = numpy.zeros_like(query_vector)
                for idx in subset:
                    resultant_vector += numpy.array(rows[idx][:-2]) - query_vector
                vector_sum_length = numpy.linalg.norm(resultant_vector)
                if vector_sum_length < min_vector_sum_length:
                    min_vector_sum_length = vector_sum_length
                    best_subset = subset
        selected_values = [rows[idx][-2] for idx in best_subset]
        weights = [1 / len(best_subset) for _ in best_subset]

        return selected_values, weights

    def _get_knn_values(self, params: Dict[str, float], k=5):
        """
        return [Si, Cu,..., value, distance] for each of the k nearest neighbors
        """
        query_point = [[params.get(element, 0) for element in self.ELEMENTS]]
        distances, indices = self.KDTREE.query(query_point, k=k, return_distance=True)
        return [[*self.composition_vectors[i], self.values[i], distances[0][dist_i]] for dist_i, i in enumerate(indices[0])]

    def query_from_file(self, filename, k=5):
        results = ""
        # check header in query file and data file matches
        with open(filename, "r") as file:
            header = file.readline().strip()
            query_elements = [item.lower() for item in header.split("\t")]
            if query_elements != self.ELEMENTS:
                print(f"Query file elements {query_elements} do not match data file elements {self.ELEMENTS}")
                raise ValueError(f"Query file elements {query_elements} do not match data file elements {self.ELEMENTS}")
            
            for line in file.readlines(): # head line has been read
                items = re.split("\t", line.strip())
                params = {query_elements[i]: float(item) for i,item in enumerate(items)}
                # print("params", params)
                results += line + self.query(params, k) + "\n"
        return results

    def output_res(self, result, output_file):
        if not os.path.exists(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as file:
            file.write(result)

    def load_data(self, filename="matQuery.pkl"):
        with gzip.GzipFile(filename, "rb") as f:
            return pickle.load(f)
