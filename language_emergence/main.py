import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))   # language_emergence/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                    # project root (where data/ lives)
sys.path.insert(0, BASE_DIR)

from configs.experiment_config import ExperimentConfig
from src.models.autoencoder import Autoencoder, trainSAE, testSAE
from src.models.dqn import DQN
from src.rl.teacher import generate_q_matrices, q_matrix_from_network
from src.evaluation.student_eval import (
    train_student, run_evaluations, train_student_with_feedback, close_the_loop
)
from src.utils.data_loading import QMatrixData
from src.utils.plotting import plot_pca_variance, plot_pca_by_label
from src.utils.generate_teacher_data import save_teacher_data

# ---------------------------------------------------------------------------
# Checkpoint / output directory helpers
# ---------------------------------------------------------------------------

def _ckpt_dir():
    d = os.path.join(BASE_DIR, 'outputs', 'checkpoints')
    os.makedirs(d, exist_ok=True)
    return d

def _fig_dir():
    d = os.path.join(BASE_DIR, 'outputs', 'figures')
    os.makedirs(d, exist_ok=True)
    return d

def _tbl_dir():
    d = os.path.join(BASE_DIR, 'outputs', 'tables')
    os.makedirs(d, exist_ok=True)
    return d

def _save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def _load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _new_autoencoder(config):
    return Autoencoder(
        in_channels=4, maze_dim=config.grid_dim,
        num_channels=10, kernel_dim=2, K=config.K
    )

def _load_autoencoder(config, device, filename):
    ae = _new_autoencoder(config)
    ae.load_state_dict(torch.load(os.path.join(_ckpt_dir(), filename), map_location=device))
    return ae

def _load_student(config, device, filename):
    std = DQN(K=config.K, n_actions=config.n_actions, device=device)
    std.load_state_dict(torch.load(os.path.join(_ckpt_dir(), filename), map_location=device))
    return std

# ---------------------------------------------------------------------------
# Stage 1: generate_teacher
# ---------------------------------------------------------------------------

def stage_generate_data(config):
    print("--- Stage 0: Generate Teacher Data PKLs ---")
    save_teacher_data(PROJECT_ROOT, config)


def stage_generate_teacher(config, device, regen=False):
    print("--- Stage 1: Teacher Q-Matrix Generation ---")
    ckpt = _ckpt_dir()

    if regen:
        config.qmat_gen = True
        print("  --regen set: Q-matrices will be regenerated from scratch")

    config.qmat_read_code = 'training_4x4'
    q_matrices, label_dict, wall_state_dict, sopt_dict = generate_q_matrices(config, device)
    _save_pkl((q_matrices, label_dict, wall_state_dict, sopt_dict),
              os.path.join(ckpt, 'teacher_train.pkl'))
    print(f"  Saved teacher_train.pkl ({len(label_dict)} tasks)")

    config.qmat_read_code = 'test_4x4'
    q_matrices_test, label_dict_test, wall_state_dict_test, sopt_dict_test = generate_q_matrices(config, device)
    _save_pkl((q_matrices_test, label_dict_test, wall_state_dict_test, sopt_dict_test),
              os.path.join(ckpt, 'teacher_test.pkl'))
    print(f"  Saved teacher_test.pkl ({len(label_dict_test)} tasks)")

# ---------------------------------------------------------------------------
# Stage 2: train_language
# ---------------------------------------------------------------------------

def stage_train_language(config, device):
    print("--- Stage 2: Language Training (SAE) ---")
    ckpt = _ckpt_dir()

    train_dataset = QMatrixData(
        q_file_path=os.path.join(BASE_DIR, 'data/teacher/q matrix dictionaries/q_matricestraining_4x4.pkl'),
        label_file_path=os.path.join(BASE_DIR, 'data/teacher/label dictionaries/q_matrices_labelstraining_4x4.pkl'),
    )

    autoencoder = _new_autoencoder(config)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.learning_rate_autoenc)

    trainSAE(autoencoder, train_dataset, config.gamma_sparse,
             optimizer, batch_size=10, epochs=config.training_epochs, device=device)

    _, _, _, all_messages, message_dict = testSAE(
        autoencoder, train_dataset, batch_size=10,
        gamma_sparse=config.gamma_sparse, device=device
    )

    torch.save(autoencoder.state_dict(), os.path.join(ckpt, 'autoencoder_initial.pt'))
    _save_pkl({'message_dict': message_dict, 'all_messages': all_messages},
              os.path.join(ckpt, 'messages_initial.pkl'))
    print("  Saved autoencoder_initial.pt + messages_initial.pkl")

# ---------------------------------------------------------------------------
# Stage 3: train_student
# ---------------------------------------------------------------------------

def stage_train_student(config, device):
    print("--- Stage 3: Baseline Student Training ---")
    ckpt = _ckpt_dir()

    q_matrices, label_dict, wall_state_dict, sopt_dict = _load_pkl(os.path.join(ckpt, 'teacher_train.pkl'))
    message_dict = _load_pkl(os.path.join(ckpt, 'messages_initial.pkl'))['message_dict']

    student = train_student(label_dict, wall_state_dict, sopt_dict, message_dict, config, device)
    torch.save(student.state_dict(), os.path.join(ckpt, 'student_baseline.pt'))

    print("\n  Baseline evaluation:")
    run_evaluations(student, label_dict, wall_state_dict, message_dict, sopt_dict, config, device)
    print("  Saved student_baseline.pt")

# ---------------------------------------------------------------------------
# Stage 4: feedback
# ---------------------------------------------------------------------------

def stage_feedback(config, device):
    print("--- Stage 4: Feedback Training ---")
    ckpt = _ckpt_dir()

    q_matrices, label_dict, wall_state_dict, sopt_dict = _load_pkl(os.path.join(ckpt, 'teacher_train.pkl'))
    message_dict = _load_pkl(os.path.join(ckpt, 'messages_initial.pkl'))['message_dict']

    autoencoder = _load_autoencoder(config, device, 'autoencoder_initial.pt')
    student     = _load_student(config, device, 'student_baseline.pt')

    autoencoder, student = train_student_with_feedback(
        label_dict, wall_state_dict, sopt_dict, q_matrices,
        autoencoder, student, config, device
    )

    message_dict_feedback = {}
    autoencoder.eval()
    with torch.no_grad():
        for task, (wall_label, init_state, goal_state) in label_dict.items():
            q_mat = q_matrices[task].unsqueeze(0).to(device)
            message, _ = autoencoder(q_mat)
            message_dict_feedback[(wall_label, init_state, goal_state)] = message.squeeze(0).detach()

    torch.save(autoencoder.state_dict(), os.path.join(ckpt, 'autoencoder_feedback.pt'))
    torch.save(student.state_dict(),     os.path.join(ckpt, 'student_feedback.pt'))
    _save_pkl(message_dict_feedback,     os.path.join(ckpt, 'messages_feedback.pkl'))

    print("\n  Feedback evaluation:")
    run_evaluations(student, label_dict, wall_state_dict, message_dict_feedback, sopt_dict, config, device)
    print("  Saved autoencoder_feedback.pt + student_feedback.pt + messages_feedback.pkl")

# ---------------------------------------------------------------------------
# Stage 5: telephone_game
# ---------------------------------------------------------------------------

def stage_telephone_game(config, device):
    print("--- Stage 5: Telephone Game ---")
    ckpt = _ckpt_dir()
    figs = _fig_dir()

    q_matrices, label_dict, wall_state_dict, sopt_dict = _load_pkl(os.path.join(ckpt, 'teacher_train.pkl'))
    message_dict          = _load_pkl(os.path.join(ckpt, 'messages_initial.pkl'))['message_dict']
    message_dict_feedback = _load_pkl(os.path.join(ckpt, 'messages_feedback.pkl'))

    autoencoder      = _load_autoencoder(config, device, 'autoencoder_feedback.pt')
    student_feedback = _load_student(config, device, 'student_feedback.pt')
    student_baseline = _load_student(config, device, 'student_baseline.pt')

    # Evaluate both baseline and feedback to populate the results table
    print("\n  Evaluating gen0 (no feedback):")
    inf0, misinf0 = run_evaluations(
        student_baseline, label_dict, wall_state_dict, message_dict, sopt_dict, config, device
    )
    print("\n  Evaluating gen1 (feedback):")
    inf1, misinf1 = run_evaluations(
        student_feedback, label_dict, wall_state_dict, message_dict_feedback, sopt_dict, config, device
    )

    # Gen 2
    print("\n=== Generation 2 ===")
    student_gen2, messages_gen2 = close_the_loop(
        student_feedback, autoencoder, label_dict, wall_state_dict,
        message_dict_feedback, sopt_dict, config, device
    )
    inf2, misinf2 = run_evaluations(
        student_gen2, label_dict, wall_state_dict, messages_gen2, sopt_dict, config, device
    )

    results = {
        'gen':         ['no feedback', 'gen1 (feedback)', 'gen2', 'gen3', 'gen4', 'gen5'],
        'informed':    [
            round(100 * sum(inf0)   / len(inf0),   2),
            round(100 * sum(inf1)   / len(inf1),   2),
            round(100 * sum(inf2)   / len(inf2),   2),
            None, None, None,
        ],
        'misinformed': [
            round(100 * sum(misinf0) / len(misinf0), 2),
            round(100 * sum(misinf1) / len(misinf1), 2),
            round(100 * sum(misinf2) / len(misinf2), 2),
            None, None, None,
        ],
    }

    all_messages_by_gen = {
        'gen1 (feedback)': list(message_dict_feedback.values()),
        'gen2':            list(messages_gen2.values()),
    }

    current_student  = student_gen2
    current_messages = messages_gen2

    for i, gen in enumerate(range(3, 6)):
        print(f"\n=== Generation {gen} ===")
        current_student, current_messages = close_the_loop(
            current_student, autoencoder, label_dict, wall_state_dict,
            current_messages, sopt_dict, config, device
        )
        inf, misinf = run_evaluations(
            current_student, label_dict, wall_state_dict, current_messages, sopt_dict, config, device
        )
        results['informed'][3 + i]    = round(100 * sum(inf)   / len(inf),   2)
        results['misinformed'][3 + i] = round(100 * sum(misinf) / len(misinf), 2)
        all_messages_by_gen[f'gen{gen}'] = list(current_messages.values())

    _save_pkl({'results': results, 'all_messages_by_gen': all_messages_by_gen},
              os.path.join(ckpt, 'telephone_results.pkl'))

    print(f"\n{'Condition':<20} {'Informed':>10} {'Misinformed':>13}")
    print("-" * 45)
    for g, inf, misinf in zip(results['gen'], results['informed'], results['misinformed']):
        print(f"{g:<20} {inf:>9.2f}% {misinf:>12.2f}%")

    x     = np.arange(len(results['gen']))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, results['informed'],    width, label='Informed',    color='steelblue')
    ax.bar(x + width/2, results['misinformed'], width, label='Misinformed', color='salmon')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Student Performance Across Telephone Game Generations')
    ax.set_xticks(x)
    ax.set_xticklabels(results['gen'], rotation=15)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(figs, 'fig_telephone_game.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved telephone_results.pkl + fig_telephone_game.png")

# ---------------------------------------------------------------------------
# Stage 6: figure3
# ---------------------------------------------------------------------------

def stage_figure3(config, device):
    from scipy.optimize import curve_fit
    from scipy.stats import f as f_dist
    import scipy.stats as sp_stats
    from sklearn.decomposition import PCA
    import torch.nn as nn

    print("--- Stage 6: Figure 3 + Table 5 ---")
    ckpt = _ckpt_dir()
    figs = _fig_dir()
    tbls = _tbl_dir()

    q_matrices, label_dict, wall_state_dict, sopt_dict = _load_pkl(os.path.join(ckpt, 'teacher_train.pkl'))
    msgs_init             = _load_pkl(os.path.join(ckpt, 'messages_initial.pkl'))
    message_dict          = msgs_init['message_dict']
    all_messages_flat     = msgs_init['all_messages']
    message_dict_feedback = _load_pkl(os.path.join(ckpt, 'messages_feedback.pkl'))
    tele                  = _load_pkl(os.path.join(ckpt, 'telephone_results.pkl'))
    all_messages_by_gen   = tele['all_messages_by_gen']

    autoencoder = _load_autoencoder(config, device, 'autoencoder_feedback.pt')
    student     = _load_student(config, device, 'student_feedback.pt')
    n_tasks     = len(label_dict)

    # --- PCA on initial SAE messages (notebook cell 9) ---
    messages_np      = torch.cat(all_messages_flat, dim=0).numpy()
    wall_labels_init = [label_dict[t][0] for t in range(n_tasks)]
    goal_labels_init = [label_dict[t][2] for t in range(n_tasks)]
    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(messages_np)
    plot_pca_variance(pca, save_name='pca_variance_initial.png')
    plot_pca_by_label(pca_result, wall_labels_init, 'initial — color: wall position')
    plot_pca_by_label(pca_result, goal_labels_init, 'initial — color: goal labels')

    # --- PCA per telephone-game generation (notebook cell 15) ---
    wall_labels = [label_dict[t][0] for t in label_dict]
    goal_labels = [label_dict[t][2] for t in label_dict]
    for gen_label, msgs in all_messages_by_gen.items():
        msgs_np = torch.stack(msgs).detach().cpu().numpy()
        pca_g   = PCA(n_components=5)
        pca_r   = pca_g.fit_transform(msgs_np)
        print(f"\n  --- {gen_label} ---")
        safe = gen_label.replace(' ', '_').replace('(', '').replace(')', '')
        plot_pca_variance(pca_g, save_name=f'pca_variance_{safe}.png')
        plot_pca_by_label(pca_r, wall_labels, f'{gen_label} — color: wall position')
        plot_pca_by_label(pca_r, goal_labels, f'{gen_label} — color: goal location')

    # --- Table 5: Critical F-values (notebook cell 16) ---
    dfn, dfd = 14, 195
    alphas   = [0.1, 0.05, 0.01, 0.005, 0.001]
    lines    = [
        "Table 5 | Critical F-values for goal groupings (dfn=14, dfd=195)",
        "{:<20} {:>8}".format("Significance (a)", "F_crit"),
        "-" * 30,
    ]
    for alpha in alphas:
        lines.append("{:<20} {:>8.2f}".format(alpha, f_dist.ppf(1 - alpha, dfn, dfd)))
    output = "\n".join(lines)
    print(f"\n{output}")
    with open(os.path.join(tbls, 'table5_critical_f.txt'), 'w') as f:
        f.write(output + "\n")
    print("  Saved table5_critical_f.txt")

    # --- Figure 3 (b, c): Topographic Similarity (notebook cell 17) ---
    norm_topo = config.norm_topo
    n_bins    = config.n_bins_topo
    w_goal    = config.goal_reward
    w_wall    = config.wall_reward
    grid_dim  = config.grid_dim

    state_coord = {i: np.array([i // grid_dim, i % grid_dim]) for i in range(grid_dim ** 2)}
    state_coord[0] = np.array([grid_dim, grid_dim])

    tasks_list = [label_dict[t] for t in range(n_tasks)]
    vwdict = {
        t: np.concatenate([w_goal * state_coord[gs], w_wall * state_coord[wi]])
        for t, (wi, _, gs) in enumerate(tasks_list)
    }
    qdict = {t: q_matrices[t].flatten().detach().numpy() for t in range(n_tasks)}

    pairs  = [(i, j) for i in range(n_tasks) for j in range(i + 1, n_tasks)]
    qdist  = np.array([np.linalg.norm(qdict[i]  - qdict[j],  ord=norm_topo) for i, j in pairs])
    vwdist = np.array([np.linalg.norm(vwdict[i] - vwdict[j], ord=norm_topo) for i, j in pairs])

    q_sorted  = sorted(range(len(pairs)), key=lambda k: qdist[k])
    vw_sorted = sorted(range(len(pairs)), key=lambda k: vwdist[k])

    msg_nofb = {t: message_dict[tuple(label_dict[t])].detach().numpy()          for t in range(n_tasks)}
    msg_fb   = {t: message_dict_feedback[tuple(label_dict[t])].detach().numpy() for t in range(n_tasks)}

    def bin_topo(msg_by_task, sorted_keys, meaning_dist):
        mdist = np.array([np.linalg.norm(msg_by_task[i] - msg_by_task[j]) for i, j in pairs])
        el    = len(sorted_keys) // n_bins
        xvals, yvals = [], []
        for k in range(n_bins):
            chunk = sorted_keys[el * k: el * (k + 1)]
            xvals.append(np.mean(meaning_dist[chunk]))
            yvals.append(np.mean(mdist[chunk]))
        return np.array(xvals), np.array(yvals)

    xq_nofb,  yq_nofb  = bin_topo(msg_nofb, q_sorted,  qdist)
    xq_fb,    yq_fb    = bin_topo(msg_fb,   q_sorted,  qdist)
    xvw_nofb, yvw_nofb = bin_topo(msg_nofb, vw_sorted, vwdist)
    xvw_fb,   yvw_fb   = bin_topo(msg_fb,   vw_sorted, vwdist)

    def linfunc(x, m, b):
        return m * x + b

    cmap_topo = plt.get_cmap("bwr")
    fig, ax   = plt.subplots(1, 2, figsize=(14, 5.6))
    for j, (xnofb, ynofb, xfb, yfb, xlabel) in enumerate([
        (xq_nofb,  yq_nofb,  xq_fb,  yq_fb,  "teacher Q-matrix distance"),
        (xvw_nofb, yvw_nofb, xvw_fb, yvw_fb, "spatial task distance"),
    ]):
        for x, y, color, label in [
            (xfb,   yfb,   cmap_topo(0.15), "with feedb."),
            (xnofb, ynofb, cmap_topo(0.99), "no feedb."),
        ]:
            popt, pcov = curve_fit(linfunc, x, y)
            m_val = round(popt[0], 3)
            m_std = round(np.sqrt(np.diag(pcov))[0], 3)
            ax[j].scatter(x, y, color=color, s=40, label=f"{label}: m={m_val}\u00b1{m_std}")
            ax[j].plot(x, linfunc(x, *popt), color=color, lw=2)
        ax[j].set_xlabel(xlabel, fontsize=20)
        ax[j].legend(fontsize=14, loc=2)
        ax[j].set_ylim(0, 1.2)
        ax[j].tick_params(width=3, length=8)
    ax[0].set_ylabel("message distance", fontsize=20)
    ax[0].set_title("teacher meaning", fontsize=22, y=1.05)
    ax[1].set_title("spatial meaning",  fontsize=22, y=1.05)
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(os.path.join(figs, 'fig3_bc_topographic_similarity.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved fig3_bc_topographic_similarity.png")

    # --- Figure 3 (d, e): Shannon Entropy Analysis (notebook cell 18) ---
    H_binlist = config.H_binlist
    softy     = nn.Softmax(dim=0)

    def get_2D_PCA_entropy(input_array, n_bins):
        pca_e = PCA(n_components=2)
        proj  = pca_e.fit_transform(input_array)
        xv, yv = proj[:, 0], proj[:, 1]
        delta = max(xv.max() - xv.min(), yv.max() - yv.min()) / 2
        xmid  = (xv.max() + xv.min()) / 2
        ymid  = (yv.max() + yv.min()) / 2
        hist, _, _ = np.histogram2d(xv, yv, bins=n_bins,
                                    range=[[xmid - delta, xmid + delta],
                                           [ymid - delta, ymid + delta]])
        hist = hist.flatten() / hist.sum()
        return sp_stats.entropy(hist)

    q_teacher_arr = np.array([q_matrices[t].flatten().detach().numpy()                        for t in range(n_tasks)])
    msg_nofb_arr  = np.array([message_dict[tuple(label_dict[t])].detach().numpy()             for t in range(n_tasks)])
    msg_fb_arr    = np.array([message_dict_feedback[tuple(label_dict[t])].detach().numpy()    for t in range(n_tasks)])

    student.eval()
    autoencoder.eval()
    p_student_list = []
    with torch.no_grad():
        for t in range(n_tasks):
            wi, _, gs  = label_dict[t]
            msg        = message_dict_feedback[tuple(label_dict[t])].to(device)
            wall_s     = wall_state_dict[wi]
            q_mat      = q_matrix_from_network(student, msg, wall_s, config, device).cpu()
            p_student_list.append(softy(q_mat.flatten()).numpy())
    p_student_arr = np.array(p_student_list)

    H_q_teacher = np.array([get_2D_PCA_entropy(q_teacher_arr, n) for n in H_binlist])
    H_msg_nofb  = np.array([get_2D_PCA_entropy(msg_nofb_arr,  n) for n in H_binlist])
    H_msg_fb    = np.array([get_2D_PCA_entropy(msg_fb_arr,    n) for n in H_binlist])
    H_p_student = np.array([get_2D_PCA_entropy(p_student_arr, n) for n in H_binlist])

    Hmax     = sp_stats.entropy([1 / n_tasks] * n_tasks)
    cmap_ent = plt.get_cmap("bwr")
    fig, ax  = plt.subplots(figsize=(9, 6))
    ax.plot([H_binlist[0], H_binlist[-1]], [Hmax, Hmax],
            ls="--", color="black", label="maximum possible entropy", lw=2)
    for H, color, marker, label in [
        (H_q_teacher, "black",        "o", "teacher Q-matrices"),
        (H_msg_fb,    cmap_ent(0.12), "D", "messages w/ feedback"),
        (H_msg_nofb,  cmap_ent(0.99), "s", "messages w/o feedback"),
        (H_p_student, cmap_ent(0.35), "D", "student action prob. matrices"),
    ]:
        ax.scatter(H_binlist, H, color=color, marker=marker, s=75, label=label, lw=2)
    ax.set_xlabel(r"bins per PC-axis ($\it{n}$)", fontsize=22)
    ax.set_ylabel("Shannon entropy", fontsize=22)
    ax.set_title("entropy at different comm. stages", fontsize=22)
    ax.set_xlim(0, 64)
    ax.set_ylim(2, 5.5)
    ax.legend(fontsize=12, loc=4)
    ax.tick_params(width=3, length=8)
    fig.tight_layout()
    plt.savefig(os.path.join(figs, 'fig3_de_entropy.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved fig3_de_entropy.png")

# ---------------------------------------------------------------------------
# Stage 7: figure4
# ---------------------------------------------------------------------------

def stage_figure4(config, device):
    import scipy.stats as sp_stats

    print("--- Stage 7: Figure 4 + Table 6 (35 runs) ---")
    ckpt = _ckpt_dir()
    figs = _fig_dir()
    tbls = _tbl_dir()

    q_matrices, label_dict, wall_state_dict, sopt_dict = _load_pkl(os.path.join(ckpt, 'teacher_train.pkl'))
    q_matrices_unk, label_dict_unk, wall_state_dict_unk, sopt_dict_unk = _load_pkl(os.path.join(ckpt, 'teacher_test.pkl'))

    train_goals_dict = {
        0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        1: [1, 4, 3, 6, 9, 12, 11, 14],
        2: [2, 5, 8, 7, 10, 13, 15],
        3: [1, 4, 5, 8, 9, 12, 13],
        4: [2, 3, 6, 7, 10, 11, 14, 15],
        5: [1, 2, 3, 4, 5, 6, 7],
        6: [8, 9, 10, 11, 12, 13, 14, 15],
    }

    N_LANGS  = 5
    N_GROUPS = 7

    def make_message_dict(ae, ld, qmats):
        ae.eval()
        mdict = {}
        with torch.no_grad():
            for task_idx, label in ld.items():
                wi, init, goal = label[0], label[1], label[2]
                q = qmats[task_idx].unsqueeze(0).to(device)
                msg, _ = ae(q)
                mdict[(wi, init, goal)] = msg.squeeze(0).detach()
        return mdict

    fig4_rates = {
        g: {cond: {s: [] for s in range(2)} for cond in range(4)}
        for g in range(N_GROUPS)
    }

    for g in range(N_GROUPS):
        train_goals   = train_goals_dict[g]
        unknown_goals = [gs for gs in range(1, 16) if gs not in train_goals]

        print(f"\n{'='*55}")
        print(f"Goal group {g}: train on {train_goals}")
        print(f"{'='*55}")

        filt_label = {t: lbl for t, lbl in label_dict.items() if lbl[2] in train_goals}
        filt_qmat  = {t: q_matrices[t] for t in filt_label}
        filt_sopt  = {t: sopt_dict[t]  for t in filt_label}

        for l in range(N_LANGS):
            print(f"\n  --- Group {g}, Language {l + 1}/{N_LANGS} ---")

            ae_run  = _new_autoencoder(config)
            std_run = DQN(K=config.K, n_actions=config.n_actions, device=device)

            ae_run, std_run = train_student_with_feedback(
                filt_label, wall_state_dict, filt_sopt, filt_qmat,
                ae_run, std_run, config, device
            )

            msg_train = make_message_dict(ae_run, label_dict,     q_matrices)
            msg_test  = make_message_dict(ae_run, label_dict_unk, q_matrices_unk)

            ld_conds = [
                ({t: lbl for t, lbl in label_dict.items()     if lbl[2] in train_goals},   wall_state_dict,     msg_train, sopt_dict),
                ({t: lbl for t, lbl in label_dict.items()     if lbl[2] in unknown_goals}, wall_state_dict,     msg_train, sopt_dict),
                ({t: lbl for t, lbl in label_dict_unk.items() if lbl[2] in train_goals},   wall_state_dict_unk, msg_test,  sopt_dict_unk),
                ({t: lbl for t, lbl in label_dict_unk.items() if lbl[2] in unknown_goals}, wall_state_dict_unk, msg_test,  sopt_dict_unk),
            ]

            for cond_idx, (ld_c, wsd, msg, so) in enumerate(ld_conds):
                if not ld_c:
                    continue
                inf, misinf = run_evaluations(std_run, ld_c, wsd, msg, so, config, device)
                fig4_rates[g][cond_idx][0].append(sum(inf)    / len(inf))
                fig4_rates[g][cond_idx][1].append(sum(misinf) / len(misinf))
                print(f"    cond {cond_idx}: inf={fig4_rates[g][cond_idx][0][-1]:.3f}  misinf={fig4_rates[g][cond_idx][1][-1]:.3f}")

    print("\nAll 35 runs complete!")
    _save_pkl(fig4_rates, os.path.join(ckpt, 'fig4_rates.pkl'))

    # Bar charts (notebook cell 20)
    cmap_fig4   = plt.get_cmap("bwr")
    cond_titles = [
        "trained goals — trained mazes",
        "unknown goals — trained mazes",
        "trained goals — unknown mazes",
        "unknown goals — unknown mazes",
    ]
    for cond_idx in range(4):
        groups = list(range(N_GROUPS)) if cond_idx in [0, 2] else list(range(1, N_GROUPS))
        avg_inf, sem_inf, avg_misinf, sem_misinf, pvals = [], [], [], [], []
        for g in groups:
            inf_rates    = fig4_rates[g][cond_idx][0]
            misinf_rates = fig4_rates[g][cond_idx][1]
            if not inf_rates:
                continue
            n = len(inf_rates)
            avg_inf.append(100 * np.mean(inf_rates))
            sem_inf.append(100 * np.std(inf_rates, ddof=1) / np.sqrt(n) if n > 1 else 0)
            avg_misinf.append(100 * np.mean(misinf_rates))
            sem_misinf.append(100 * np.std(misinf_rates, ddof=1) / np.sqrt(n) if n > 1 else 0)
            pvals.append(sp_stats.ttest_ind(inf_rates, misinf_rates).pvalue if n > 1 else 1.0)

        x     = np.arange(len(groups))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(groups)), 5))
        ax.bar(x - width/2, avg_inf,    width, color=cmap_fig4(0.15), label="informed")
        ax.bar(x + width/2, avg_misinf, width, color=cmap_fig4(0.85), label="misinformed")
        ax.errorbar(x - width/2, avg_inf,    yerr=sem_inf,    fmt="none", ecolor="black", capsize=4, elinewidth=2)
        ax.errorbar(x + width/2, avg_misinf, yerr=sem_misinf, fmt="none", ecolor="black", capsize=4, elinewidth=2)
        for i, (yi, ym, p) in enumerate(zip(avg_inf, avg_misinf, pvals)):
            if p < 0.05 and yi > ym:
                y_star = max(yi + sem_inf[i], ym + sem_misinf[i]) + 3
                ax.annotate("*", xy=(x[i], y_star), ha="center", fontsize=16)
        ax.set_ylim(0, 120)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylabel("tasks solved (%)", fontsize=14)
        ax.set_xlabel("goal locations trained", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in groups])
        ax.set_title(cond_titles[cond_idx], fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(figs, f'fig4_cond{cond_idx}.png'), bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Saved fig4_cond{cond_idx}.png")

    # Table 6: Bonferroni-corrected t-tests (notebook cell 21)
    cond_names = [
        "trained goals, trained mazes",
        "unknown goals, trained mazes",
        "trained goals, unknown mazes",
        "unknown goals, unknown mazes",
    ]
    all_pvals, all_labels = [], []
    for cond_idx in range(4):
        groups = list(range(N_GROUPS)) if cond_idx in [0, 2] else list(range(1, N_GROUPS))
        for g in groups:
            inf_rates    = fig4_rates[g][cond_idx][0]
            misinf_rates = fig4_rates[g][cond_idx][1]
            if len(inf_rates) < 2:
                continue
            all_pvals.append(sp_stats.ttest_ind(inf_rates, misinf_rates).pvalue)
            all_labels.append((cond_idx, g))

    n_tests     = len(all_pvals)
    p_corrected = [min(p * n_tests, 1.0) for p in all_pvals]
    col_header  = "{:<35} {:<6} {:<10} {:<11} {}".format("Condition", "Group", "p (raw)", "p (Bonf.)", "sig")
    lines = [
        f"Table 6 | Bonferroni-corrected t-tests: informed vs. misinformed (N={n_tests} comparisons)",
        col_header,
        "-" * 67,
    ]
    for (cond_idx, g), p_raw, p_corr in zip(all_labels, all_pvals, p_corrected):
        sig = "*" if p_corr < 0.05 else ""
        lines.append("{:<35} {:<6} {:<10.4f} {:<11.4f} {}".format(
            cond_names[cond_idx], g, p_raw, p_corr, sig))
    output = "\n".join(lines)
    print(f"\n{output}")
    with open(os.path.join(tbls, 'table6_bonferroni.txt'), 'w') as f:
        f.write(output + "\n")
    print("  Saved table6_bonferroni.txt")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Language Emergence Execution Pipeline")
    parser.add_argument(
        '--stage', type=str, required=True,
        choices=['all', 'generate_data', 'generate_teacher', 'train_language',
                 'train_student', 'feedback', 'telephone_game', 'figure3', 'figure4'],
        help='Which stage of the pipeline to run'
    )
    parser.add_argument(
        '--regen', action='store_true',
        help='Regenerate Q-matrices from scratch instead of loading existing PKLs'
    )
    args   = parser.parse_args()
    config = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stage_fns = {
        'generate_data':    lambda: stage_generate_data(config),
        'generate_teacher': lambda: stage_generate_teacher(config, device, regen=args.regen),
        'train_language':   lambda: stage_train_language(config, device),
        'train_student':    lambda: stage_train_student(config, device),
        'feedback':         lambda: stage_feedback(config, device),
        'telephone_game':   lambda: stage_telephone_game(config, device),
        'figure3':          lambda: stage_figure3(config, device),
        'figure4':          lambda: stage_figure4(config, device),
    }

    if args.stage == 'all':
        for fn in stage_fns.values():
            print(f"\n{'='*60}")
            fn()
    else:
        stage_fns[args.stage]()


if __name__ == "__main__":
    main()
