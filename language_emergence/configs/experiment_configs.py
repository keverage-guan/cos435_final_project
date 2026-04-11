import math
import numpy as np

class ExperimentConfig:
    def __init__(self):
        self.data_dir = "outputs"
        self.language_code = "nonlinear_goallocs0_zeta5_language2"
        self.language_code_closingloop = "nonlinear_goallocs0_zeta5_language0"
        self.qmat_read_code = "training_4x4"
        
        # environment (gridworld) parameters
        self.grid_dim = 4
        self.n_actions = 4
        self.student_init = 0
        self.lava = False
        
        self.step_reward = -0.1
        self.goal_reward = 2.0
        self.wall_reward = -0.5
        
        # teacher parameters (q-matrix)
        self.qmat_gen = False
        self.accuracy = 20
        self.max_attempts = 3
        self.gamma_bellman = 0.99
        self.L = 50
        self.lr_teacher = 3e-3
        
        # auto encoder parameters (language generation)
        self.language_gen = False
        self.K = 5
        self.kappa = 1/500
        self.learning_rate_autoenc = 5e-4
        self.training_epochs = 50
        self.zeta_std = 5
        
        # evaluation parameters (student performance)
        self.student_evaluate = False
        self.save_rates = True
        self.chuck_out = False
        
        self.folder_stepfactor_plots = "nonlinear_goallocs0_zeta5_factor1"
        self.language_nr_stepfactor_plots = 5
        self.save_stepfactor_plots = False
        
        self.language_nr_goalloc_plots = 5
        self.stepfactor_goalloc_plots = 2
        self.rdrate_goalloc_plots = 0.25
        self.save_goalloc_plots = False
        
        # plotting and analysis parameters
        self.zeta_lossplot = 5
        self.epskip = 50
        self.save_autoenc_lossplots = False
        
        self.do_tsne_message_plots = True
        self.do_umap_message_plots = True
        self.save_message_plots = False
        
        self.closingloop_nonlinear_ae = True
        self.closingloop_nonlinear_std = True
        self.save_closingloop_plots = False
        
        self.norm_topo = 2
        self.n_bins_topo = 20

        # dicts and lists
        self.wall_state_dict = {0: []}
        self.plot_worlds_single = [6]
        self.stepfactor_list_plots = [1, 1.5, 2, 2.5, 3, 3.5, 4]
        self.rdrates_list_plots = []
        self.folders_goalloc_plots = [f"nonlinear_goallocs{i}_zeta5_factor1" for i in range(7)]
        self.plot_worlds = range(16)
        self.goal_groups_plots = np.array([0, 1, 2, 3, 4, 5, 6])
        self.H_binlist = np.linspace(8, 62, 10).astype(int)
        
        # derived variables
        self.data_shape = (self.n_actions, self.grid_dim, self.grid_dim)
        
        self.gamma_sparse = (1/20) * math.sqrt((self.grid_dim**2 * self.n_actions) / self.K)
        
        self.nonlinear_ae_plots = "nonlinear" in self.language_code
        self.nonlinear_std_plots = ("nonlinear" in self.language_code) or ("linear_ae" in self.language_code)