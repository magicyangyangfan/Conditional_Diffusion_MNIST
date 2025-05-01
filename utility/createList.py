import os
from os import walk
import pandas as pd
from phasePrediction2 import phasePredictionLite
import random

dict_AlMgSiCu ={
        #       Si','Fe','Cu','Mn', 'Mg', 'Ni'
        'A356': {"Si": 0.07, "Cu": 0.0025, "Mg": 0.003, "Al": 0.9245},
        'A360': {"Si": 0.095, "Cu": 0.001, "Mg": 0.005, "Al": 0.899},
        'A369': {"Si": 0.115, "Cu": 0.005, "Mg": 0.004, "Al": 0.876},
        'A339': {"Si": 0.12, "Cu": 0.02, "Mg": 0.01, "Al": 0.85},
        'A393': {"Si": 0.22, "Cu": 0.009, "Mg": 0.001, "Al": 0.77},
        'A355': {"Si": 0.05, "Cu": 0.0125, "Mg": 0.005, "Al": 0.9325},
        'A308': {"Si": 0.055, "Cu": 0.045, "Mg": 0.001, "Al": 0.899},
        'A319': {"Si": 0.06, "Cu": 0.04, "Mg": 0.0055, "Al": 0.8945},
        'A332': {"Si": 0.095, "Cu": 0.03, "Mg": 0.021, "Al": 0.854},
        'SandTH3': 0.017,
        'SandTH1': 0.07,
        'PermTH2': 0.4,
        'PermTH1': 1.0,
        'N': 0,
        'M': 1
        }

class createAlSiMgCuPhaseList():
    def __init__(self, pkl_path, img_dir, n_validation, use_random_pick=False):
        self.phasePredictor = phasePredictionLite(pkl_path)
        self.img_dir=img_dir
        self.df_training = pd.DataFrame()
        self.df_validation = pd.DataFrame()
        self.df_temp = None
        self.n_validation = n_validation
        self.use_random_pick = use_random_pick

    def setList(self, save_dir, file_name: str):
        #self.df=pd.DataFrame(columns=['file name','p_FCC_A1#1','p_MG2SI_C1#1','e_FCC_A1#1','e_DIAMOND_A4#1', 'e_Q_ALCUMGSI#1', 'CoolingRate','Midification'],index=None)
        # dir_list is the full list of dir
        dir_list = [name for name in os.listdir(self.img_dir) if os.path.isdir(os.path.join(self.img_dir, name))]
        assert len(dir_list) > self.n_validation
        if self.use_random_pick:
            validation_dirs = random.sample(dir_list, self.n_validation)
            training_dirs = list(set(dir_list) - set(validation_dirs))
        else:
            validation_dirs = []
            training_dirs = dir_list
        
        # Process validation directories
        self.df_validation = self._generate_df_each_dir(validation_dirs, self.df_validation)
        # Process training directories (remaining directories)
        self.df_training = self._generate_df_each_dir(training_dirs, self.df_training)
        
        #resort columnn sequence to put file name, modifiation, cooling rat front
        self.df_training = self._sort_df(self.df_training)
        if validation_dirs:
            self.df_validation = self._sort_df(self.df_validation) 
    
        # saving files
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        training_file_path = os.path.join(save_dir, 'training' + file_name + '.txt')
        self.df_training.to_csv(training_file_path)
        validation_file_path = os.path.join(save_dir,'validation' + file_name + '.txt')
        self.df_validation.to_csv(validation_file_path)
        return
    
    def _sort_df(self, df):
        df = df.reindex(sorted(df.columns), axis=1)
        df_cols = list(df.columns)
        # resort columnn sequence to put file name, modification, cooling rate front
        file_col = 'file_name'
        modification_col = 'modification'
        coolin_col = 'cooling_rate'
        df_cols.remove(file_col)
        df_cols.remove(modification_col)
        df_cols.remove(coolin_col)
        df_cols.insert(0,file_col)
        df_cols.insert(1,modification_col)
        df_cols.insert(2,coolin_col)
        df = df[df_cols]
        return df


    def set_each_list(self,save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        dir_list = [name for name in os.listdir(self.img_dir) if os.path.isdir(os.path.join(self.img_dir, name))]
        for each_dir in dir_list:
            self.df_temp = pd.DataFrame()#intialize
            dir_full_path = os.path.join(self.img_dir, each_dir)
            files = [name for name in os.listdir(dir_full_path) if os.path.isfile(os.path.join(dir_full_path, name))]
            for each_file in files:
                sorted_dict = self._parseFileName(dir_full_path, each_file)
                self.df_temp =  pd.concat([self.df_temp, pd.DataFrame([sorted_dict])],ignore_index=True)
            self.df_temp = self.df_temp.reindex(sorted(self.df_temp.columns), axis=1)
            save_path = os.path.join(save_dir,each_dir)
            self.df_temp.to_csv(save_path)
        return


    def _generate_df_each_dir(self, dirs, df):
        for each_dir in dirs:
            dir_full_path = os.path.join(self.img_dir, each_dir)
            files = [name for name in os.listdir(dir_full_path) if os.path.isfile(os.path.join(dir_full_path, name))]
            for each_file in files:
                sorted_dict = self._parseFileName(dir_full_path, each_file)
                if sorted_dict is not None:
                    df = pd.concat([df, pd.DataFrame([sorted_dict])], ignore_index=True)
        return df

    def _parseFileName(self, root, file_name):
        #A356SandTH3NX100 or A356SandTH3N (len==12)
        file_name_no_extension = os.path.splitext(os.path.basename(file_name))[0] #can remove this

        composition=dict_AlMgSiCu[file_name_no_extension[:4]]
        phases = self.phasePredictor.predict(composition)

        sorted_dict = {}
        for phase, value in phases.items():
            if phase == "FCC_A1" or phase == "S_PHASE" or phase== 'T_PHASE': continue
            sorted_dict[phase] = round(value, 4)

        sorted_dict['file_name']=str(os.path.join(root,file_name))
        cooling_rate = dict_AlMgSiCu[file_name_no_extension[4:11]]
        sorted_dict['cooling_rate']=cooling_rate
        modification = dict_AlMgSiCu[file_name_no_extension[11]]
        sorted_dict['modification']=modification
        return sorted_dict

if __name__ == "__main__":
    pkl_path = '/home/yangyang/Projects/Conditional_Diffusion_MNIST/matQuery/database/AlSiMgCuScheil.pkl'
    img_data='/home/yangyang/Projects/Conditional_Diffusion_MNIST/data/TrainingData_ImgSize128_Magnification500_Rotation'
    save_dir='/home/yangyang/Projects/Conditional_Diffusion_MNIST/data/'
    listtest=createAlSiMgCuPhaseList(pkl_path, img_data, n_validation=0, use_random_pick=False)
    for i in range(1):
        file_name_index = str(i)
        listtest.setList(save_dir, file_name_index)