from matQuery.matQuery2 import matQuery
import json

class phasePredictionLite():
    def __init__(self, pkl_file):
        self.mat_query = matQuery(pkl_file)

    def predict(self,composition, k=5):
        total_phases = json.loads(self.query(composition, k))
        # print(total_phases)
        return total_phases
    
    def query(self, composition, k):
        return self.mat_query.query(composition, k)
    

# #testing below:
pkl_path = '/home/yangyang/Projects/AlumGAN/matQuery/database/AlSiMgCu.pkl'
composition_1 = {"Si": 0.07, "Cu": 0.0025, "Mg": 0.001, "Al": 0.9265}
composition_2 = {"Si": 0.08, "Cu": 0.0025, "Mg": 0.003, "Al": 0.9145}
# composition_2 = {"Si": 0.095, "Cu": 0.001, "Mg": 0.005, "Al": 0.899}
# composition_3 = {"Si": 0.115, "Cu": 0.005, "Mg": 0.004, "Al": 0.876}
# composition_4 = {"Si": 0.12, "Cu": 0.02, "Mg": 0.01, "Al": 0.85}
# composition_5 = {"Si": 0.22, "Cu": 0.009, "Mg": 0.001, "Al": 0.77}
# composition_6 = {"Si": 0.05, "Cu": 0.0125, "Mg": 0.005, "Al": 0.9325}
# composition_7 = {"Si": 0.055, "Cu": 0.045, "Mg": 0.001, "Al": 0.899}
# composition_8 = {"Si": 0.06, "Cu": 0.04, "Mg": 0.0055, "Al": 0.8945}
# composition_9 = {"Si": 0.095, "Cu": 0.03, "Mg": 0.021, "Al": 0.854}
# # phase = phasePrediction(pkl_path, composition).get_above_solidus_phase()
# # phase = phasePrediction(pkl_path, composition).get_solidus_phase()
# # phase = phasePrediction(pkl_path, composition).get_eutectic_phases_vol_frac()
# # phase = phasePrediction(pkl_path, composition).get_primary_phases_vol_frac()
# phase = phasePrediction(pkl_path)
# phases = phase.predict(composition_2)
# print(phases)
# # phase.predict(composition_2)
# # phase.predict(composition_3)
# # phase.predict(composition_4)
# # phase.predict(composition_5)
# # phase.predict(composition_6)
# # phase.predict(composition_7)
# # phase.predict(composition_8)
# # phase.predict(composition_9)

# print(phases)
# print(phases[1])
# Initialize the output dictionary
# sorted_dict = {}

# # # Iterate over the data
# for entry in phases:
#     for phase, phase_data in entry.items():
#         for material, value in phase_data.items():
#             # Create the new key format
#             new_key = f"{phase}_{material}"
#             # Add the new key-value pair to the dictionary
#             sorted_dict[new_key] = round(value, 4)
# print(sorted_dict)
  
