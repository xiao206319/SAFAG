from __future__ import print_function

import absl.flags as flags


# Dataset Settings
flags.DEFINE_string('dataset_path', '/media/ubuntu/9cd6fab2-42b6-40e7-9260-a20c62d641c1/yxy/test_third', 'Root path of the processed GAPart dataset.')
flags.DEFINE_string('gapart', 'all', 'GAPart category to train/test, or all.')
flags.DEFINE_integer('n_points', 1024, 'Number of input points for each sample.')
flags.DEFINE_integer('traindata_size', 8406, 'Total number of training samples.')
flags.DEFINE_integer('testdata_size_intra', 2074, 'Total number of intra-category testing samples.')
flags.DEFINE_integer('testdata_size_inter', 4754, 'Total number of inter-category testing samples.')

# Model Architecture Settings
flags.DEFINE_integer('obj_c', 10, 'Number of GAPart object categories.')
flags.DEFINE_integer('feat_c_R', 1290, 'Input feature dimension of the rotation head.')
flags.DEFINE_integer('R_c', 4, 'Output dimension of the rotation head, namely quaternion dimension.')
flags.DEFINE_integer('feat_c_ts', 1293, 'Input feature dimension of the translation-scale head.')
flags.DEFINE_integer('Ts_c', 6, 'Output dimension of translation-scale head.')
flags.DEFINE_integer('feat_face', 768, 'Input feature dimension of the face reconstruction branch.')
flags.DEFINE_integer('face_recon_c', 6 * 5, 'Output channels of face reconstruction branch.')
flags.DEFINE_integer('gcn_sup_num', 7, 'Support number used in GCN modules.')
flags.DEFINE_integer('gcn_n_num', 10, 'Neighbor number used in GCN modules.')


# Training Settings
flags.DEFINE_integer('train', 1, '1 for training mode, 0 for evaluation/inference mode.')
flags.DEFINE_integer('num_workers', 0, 'Number of DataLoader workers.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_integer('total_epoch', 400, 'Total number of training epochs.')
flags.DEFINE_integer('warm_up_epoch', 10, 'Number of warm-up epochs.')
flags.DEFINE_integer('train_steps', 1000, 'Number of training iterations per epoch.')
flags.DEFINE_integer('accumulate', 1, 'Gradient accumulation steps.')

# Checkpoint / Logging Settings
flags.DEFINE_string('model_save', '/home/chenwenxiao/code/modelsave_all', 'Directory for saving checkpoints, logs, and TensorBoard files.')
flags.DEFINE_integer('resume', 0, 'Whether to resume training from checkpoint.')
flags.DEFINE_string('resume_model', '', 'Path of checkpoint used for resuming training.')
flags.DEFINE_integer('resume_point', 300, 'Epoch index to resume from.')
flags.DEFINE_string('test_model', '', 'Path of checkpoint used for test-only evaluation.')

# Main Loss Weights
flags.DEFINE_float('recon_w', 8.0, 'Weight for reconstruction loss.')
flags.DEFINE_float('rot_w', 8.0, 'Weight for rotation loss.')
flags.DEFINE_float('trans_w', 8.0, 'Weight for translation loss.')

# Optimizer / Learning Rate Scheduler Settings
flags.DEFINE_float('lr', 1e-4, 'Base learning rate.')
flags.DEFINE_string('lr_scheduler_name', 'flat_and_anneal', 'Learning rate scheduler name.')
flags.DEFINE_string('anneal_method', 'cosine', 'Annealing method.')
flags.DEFINE_float('anneal_point', 0.72, 'Annealing start point.')
flags.DEFINE_string('optimizer_type', 'Ranger', 'Optimizer type.')
flags.DEFINE_float('weight_decay', 0.0, 'Weight decay.')
flags.DEFINE_float('warmup_factor', 0.001, 'Warm-up factor.')
flags.DEFINE_integer('warmup_iters', 1000, 'Number of warm-up iterations.')
flags.DEFINE_string('warmup_method', 'linear', 'Warm-up method.')
flags.DEFINE_float('gamma', 0.1, 'Learning rate decay gamma.')
flags.DEFINE_float('poly_power', 0.9, 'Power used by polynomial learning rate decay.')

# Face Reconstruction Loss Weights
flags.DEFINE_float('recon_n_w', 3.0, 'Weight for normal reconstruction loss.')
flags.DEFINE_float('recon_d_w', 3.0, 'Weight for distance reconstruction loss.')
flags.DEFINE_float('recon_v_w', 1.0, 'Weight for voting reconstruction loss.')
flags.DEFINE_float('recon_s_w', 0.3, 'Weight for sampling reconstruction loss.')
flags.DEFINE_float('recon_f_w', 1.0, 'Weight for face confidence reconstruction loss.')
flags.DEFINE_float('recon_bb_r_w', 1.0, 'Weight for bounding-box rotation reconstruction loss.')
flags.DEFINE_float('recon_bb_t_w', 1.0, 'Weight for bounding-box translation reconstruction loss.')
flags.DEFINE_float('recon_bb_s_w', 1.0, 'Weight for bounding-box scale reconstruction loss.')
flags.DEFINE_float('recon_bb_self_w', 1.0, 'Weight for bounding-box self-consistency loss.')
